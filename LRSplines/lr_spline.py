import typing
from typing import List

import matplotlib.patches as plp
import matplotlib.pyplot as plt
import numpy as np

from LRSplines.aux_split_functions import split_single_basis_function
from LRSplines.b_spline import BSpline, _find_knot_interval
from LRSplines.element import Element
from LRSplines.meshline import Meshline

Vector = typing.Union[typing.List['float'], np.ndarray]


def _at_end(knots, index):
    return abs(knots[-1] - knots[index]) < 1.0E-14


def init_tensor_product_LR_spline(d1: int, d2: int, ku: Vector, kv: Vector) -> 'LRSpline':
    """
    Initializes an LR spline at the tensor product level of bidegree (d1, d2).

    :param d1: first component degree
    :param d2: second component degree
    :param ku: knots in u_direction
    :param kv: knots in v_direction
    :return: corresponding LR_spline
    """

    elements = []
    basis = []
    meshlines = []

    unique_ku = np.unique(ku)
    unique_kv = np.unique(kv)

    for i in range(len(unique_ku) - 1):
        for j in range(len(unique_kv) - 1):
            elements.append(Element(unique_ku[i], unique_kv[j], unique_ku[i + 1], unique_kv[j + 1]))

    for i in range(len(ku) - d1 - 1):
        for j in range(len(kv) - d2 - 1):
            end_u = _at_end(ku, i + d1 + 1)
            end_v = _at_end(kv, j + d2 + 1)

            # TODO: This only works if the knot vectors are p+1-extended.

            north = j == len(kv) - d2 - 2
            south = j == 0
            east = i == len(ku) - d1 - 2
            west = i == 0

            basis.append(BSpline(d1, d2, ku[i: i + d1 + 2], kv[j: j + d2 + 2], end_u=end_u, end_v=end_v, north=north,
                                 south=south, east=east, west=west))

    for b in basis:
        for e in elements:
            if b.add_to_support_if_intersects(e):
                e.add_supported_b_spline(b)

    for i in range(len(unique_ku)):
        for j in range(len(unique_kv) - 1):
            new_m = Meshline(start=unique_kv[j], stop=unique_kv[j + 1], constant_value=unique_ku[i], axis=0)
            new_m.set_multiplicity(ku)
            meshlines.append(new_m)
    for i in range(len(unique_kv)):
        for j in range(len(unique_ku) - 1):
            new_m = Meshline(start=unique_ku[j], stop=unique_ku[j + 1], constant_value=unique_kv[i], axis=1)
            new_m.set_multiplicity(kv)
            meshlines.append(new_m)

    u_range = [ku[0], ku[-1]]
    v_range = [kv[0], kv[-1]]

    return LRSpline(elements, basis, meshlines, u_range, v_range, unique_ku, unique_kv)


class LRSpline(object):
    """
    Represents a LRSpline, which is a tuple (M, S), where M is a mesh and S is a set of basis functions
    defined on M.
    """

    def __init__(self, mesh: List['Element'], basis: List['BSpline'], meshlines: List['Meshline'], u_range=None,
                 v_range=None, unique_global_knots_u=None, unique_global_knots_v=None) -> None:
        """
        Initialize an LR Spline with associated set of elements, and set of basis functions.

        :param range1:
        :param range2:
        :param mesh: elements constituting the LR mesh
        :param basis: basis functions defined over the LR mesh
        """
        self.global_knots_u = np.asarray(unique_global_knots_u, dtype=np.float64)
        self.global_knots_v = np.asarray(unique_global_knots_v, dtype=np.float64)
        self.M = mesh
        self.S = basis
        self.meshlines = meshlines
        self.u_range = u_range
        self.v_range = v_range
        self.last_element = None
        self._element_cache()
        self.update_global_indices()

    def refine_by_element_full(self, e: Element) -> None:
        """
        Refines the LRSpline by finding and inserting a meshline that ensures that all supported BSplines on the
        given element will be split by the refinement.

        :param e: element to refine
        """
        raise NotImplementedError('LRSpline.{} is not implemented yet'.format(self.refine_by_element_full.__name__))

    def refine_by_element_minimal(self, e: Element) -> None:
        """
        Refines the LRSpline by finding and inserting the smallest possible meshline that splits the support of at
        least one BSpline.

        :param e: element to refine
        """
        for i in range(2):
            m = self.get_minimal_span_meshline(e, i)
            self.insert_line(m)

    @staticmethod
    def get_minimal_span_meshline(e: Element, axis) -> Meshline:
        """
        Finds the shortest possible meshline in direction prescribed by axis that splits at least one supported
        B-spline on the element.

        :param e: element to refine by
        :param axis: direction to look for split, 0 vertical, 1 horizontal
        :return: minimal span meshline
        """

        smallest_start = None
        smallest_stop = None
        for basis in e.supported_b_splines:
            k = basis.knots_v if axis == 0 else basis.knots_u
            current_length = abs(k[-1] - k[0])
            if smallest_start is None or smallest_stop is None or current_length < (smallest_stop - smallest_start):
                smallest_start = k[0]
                smallest_stop = k[-1]

        constant_value = e.midpoint[axis]

        new_meshline = Meshline(smallest_start, smallest_stop, constant_value, axis)
        return new_meshline

    @staticmethod
    def get_full_span_meshline(e: Element, axis) -> Meshline:
        """
        Finds the meshline in direction prescribed by the axis that splits all the supported B-splines on the element.
        :param e: element to refine by
        :param axis: direction to look for split, 0 vertical, 1 horizontal
        :return: full span meshline
        """
        longest_start = None
        longest_stop = None
        for basis in e.supported_b_splines:
            k = basis.knots_v if axis == 0 else basis.knots_u
            current_length = abs(k[-1] - k[0])
            if longest_start is None or longest_stop is None or current_length > (longest_stop - longest_start):
                longest_start = k[0]
                longest_stop = k[-1]
        constant_value = e.midpoint[axis]
        return Meshline(longest_start, longest_stop, constant_value, axis)

    def insert_line(self, meshline: Meshline, debug=False) -> None:
        """
        Inserts a line in the mesh, splitting where necessary.
        Follows a four step procedure:

            Step 1: Test all BSplines against the new meshline, and if the meshline traverses the support, split the
            BSpline into B1 and B2. For both B1 and B2, check whether they are already in the set of previous
            BSplines. If they are not, add them to the list of new functions. Add the function that was split to the
            list of functions to remove.

            Step 2: Test all the new B-splines against all the meshlines already present in the mesh. They might have
            to be split further.

            Step 3: Check all elements of the mesh, and make sure that any previous elements traversed by the new
            meshline are split accordingly.

            Step 4: Make sure that all elements keep track of the basis functions they support, and that all basis
            functions keep track of the elements that support them.

        :param meshline: meshline to insert
        """

        # step 0
        # merge any existing meshlines, if the meshline already exists, we are done and can return early.
        meshline_already_exists, meshline = self.merge_meshlines(meshline)

        if meshline_already_exists:
            return

        # update the list of global tensorproduct knots
        if meshline.axis == 0:
            i = np.searchsorted(self.global_knots_u, meshline.constant_value, 'left')
            self.global_knots_u = np.insert(self.global_knots_u, i, meshline.constant_value)
        elif meshline.axis == 1:
            i = np.searchsorted(self.global_knots_v, meshline.constant_value, 'left')
            self.global_knots_v = np.insert(self.global_knots_v, i, meshline.constant_value)

        # step 1
        # split B-splines against new meshline
        new_functions = []
        functions_to_remove = []
        for basis in self.S:
            if meshline.splits_basis(basis):
                if meshline.number_of_knots_contained(basis) < meshline.multiplicity:
                    self.local_split(basis, meshline, functions_to_remove, new_functions)

        purged_S = [s for s in self.S if s not in functions_to_remove]

        self.S = purged_S
        # step 2
        # split new B-splines against old meshlines
        self.meshlines.append(meshline)

        # for basis in new_functions:
        while len(new_functions) > 0:
            basis = new_functions.pop()
            split_more = False
            for m in self.meshlines:
                if m.splits_basis(basis):
                    if m.number_of_knots_contained(basis) < m.multiplicity:
                        split_more = True
                        self.local_split(basis, m, functions_to_remove, new_functions)
                        break
            if not split_more:
                self.S.append(basis)

        # step 3
        # split all marked elements against new meshline
        new_elements = []
        for element in self.M:
            if meshline.splits_element(element):
                new_elements.append(element.split(axis=meshline.axis, split_value=meshline.constant_value))

        self.M += new_elements

        # step 4
        # clean up, make sure all basis functions points to correct elements
        # make sure all elements point to correct basis functions
        # TODO: This implementation is preliminary, and possibly very slow.
        for element in self.M:
            element.supported_b_splines = []
        for basis in self.S:
            basis.elements_of_support = []
            for element in self.M:
                if basis.add_to_support_if_intersects(element):
                    element.add_supported_b_spline(basis)

        # invalidate the element cache
        self.element_cache = None
        self.update_global_indices()

    def local_split(self, basis, m, functions_to_remove, new_functions):
        b1, b2 = split_single_basis_function(m, basis)
        if self.contains_basis_function(b1):
            self._update_old_basis_function(b1, self.S)
        elif b1 in new_functions:
            self._update_old_basis_function(b1, new_functions)
        else:
            new_functions.append(b1)
        if self.contains_basis_function(b2):
            self._update_old_basis_function(b2, self.S)
        elif b2 in new_functions:
            self._update_old_basis_function(b2, new_functions)
        else:
            new_functions.append(b2)
        functions_to_remove.append(basis)

    @staticmethod
    def _update_old_basis_function(new_basis, basis_list) -> None:
        """
        Updates the basis function corresponding to b1 with new weights and coefficients, dependent on
        the basis that was split, and the new basis function.

        :param new_basis: the `new basis` originating from splitting `original_basis`, which is already present in self.S
        :return:
        """
        i = basis_list.index(new_basis)
        basis_list[i].update_weights(new_basis)

    def contains_basis_function(self, B: BSpline) -> bool:
        """
        Returns true if B is found in self.S

        :param B: BSpline to find
        :return: true or false
        """

        for b in self.S:
            if b == B:
                return True

        return False

    def contains_element(self, element: 'Element') -> bool:
        """
        Returns true if element is found in self.M

        :param element: element to check
        :return: true or false
        """

        for e in self.M:
            if e == element:
                return True
        return False

    def __call__(self, u, v):
        """
        Evaluates the LRSpline at the point (u, v)

        :param u: first component
        :param v: secont component
        :return: L(u, v)
        """

        e = self.find_element_containing_point(u, v)

        total = 0
        for b in e.supported_b_splines:
            total += b.coefficient * b(u, v)
        return total

    def find_element_containing_point(self, u, v):
        if self.last_element and self.last_element.contains(u, v):
            return self.last_element
        for e in self.M:
            if e.contains(u, v):
                break
        else:
            raise ValueError('({}, {}) is not in the domain'.format(u, v))
        self.last_element = e
        return e

    def merge_meshlines(self, meshline: Meshline) -> typing.Tuple[bool, Meshline]:
        """
        Tests the meshline against all currently stored meshlines, and combines, updates and deletes
        meshlines as needed. Returns true if the meshline is already in the list of previous meshlines.
        There are three cases:

            1. The new meshline overlaps with a previous mesh line, but is not contained by the previous one.
            2. The new meshline is completely contained in a previous mesh line, (may in fact be equal)
            3. The new meshline is completely disjoint from all other meshlines.

        :param meshline: meshline to test against previous meshlines.
        :return: true if meshline was previously found, false otherwise.
        """
        tol = 1.0e-14
        meshlines_to_remove = []

        for old_meshline in self.meshlines:
            if not old_meshline._similar(meshline):
                # the two meshlines are not comparable, continue.
                continue
            if meshline == old_meshline:
                # meshline already exists, no point in continuing
                return True, meshline

            if old_meshline.contains(meshline):
                # meshline is completely contained in the old meshline

                if meshline.multiplicity > old_meshline.multiplicity:
                    if abs(old_meshline.start - meshline.start) < tol and abs(old_meshline.stop - meshline.stop) < tol:
                        # if the new multiplicity is greater than the old one, and the endpoints coincide, keep the
                        # new line and remove the old line.
                        meshlines_to_remove.append(old_meshline)

                elif old_meshline.multiplicity >= meshline.multiplicity:
                    return True, old_meshline

            elif old_meshline.overlaps(meshline):
                if old_meshline.multiplicity < meshline.multiplicity:
                    if old_meshline.start > meshline.start:
                        old_meshline.start = meshline.start
                    if old_meshline.stop < meshline.stop:
                        old_meshline.stop = meshline.stop
                elif old_meshline.multiplicity > meshline.multiplicity:
                    if old_meshline.start < meshline.start:
                        meshline.start = old_meshline.start
                    if old_meshline.stop > meshline.stop:
                        meshline.stop = old_meshline.stop
                else:
                    if old_meshline.start < meshline.start:
                        meshline.start = old_meshline.start
                    if old_meshline.stop > meshline.stop:
                        meshline.stop = old_meshline.stop

                    meshlines_to_remove.append(old_meshline)

        for old_meshline in meshlines_to_remove:
            self.meshlines.remove(old_meshline)
        return False, meshline

    def visualize_mesh(self, multiplicity=True, overloading=True, text=True, relative=True, filename=None,
                       color=False, title=True, axes=False) -> None:
        """
        Plots the LR-mesh.
        """
        fig = plt.figure()
        axs = fig.add_subplot(1, 1, 1)
        for m in self.meshlines:
            x = (m.start, m.stop)
            y = (m.constant_value, m.constant_value)
            if m.axis == 0:
                axs.plot(y, x, color='black', linewidth=0.5)
            else:
                axs.plot(x, y, color='black', linewidth=0.5)
            if multiplicity:
                axs.text(m.midpoint[0], m.midpoint[1], '{}'.format(m.multiplicity),
                         bbox=dict(facecolor='white', alpha=1), ha='center', va='center')
        for m in self.M:
            w = m.u_max - m.u_min
            h = m.v_max - m.v_min

            if overloading:
                if m.is_overloaded():
                    axs.add_patch(plp.Rectangle((m.u_min, m.v_min), w, h, fill=True, color='red' if color else 'black',
                                                alpha=0.2))
                else:
                    axs.add_patch(
                        plp.Rectangle((m.u_min, m.v_min), w, h, fill=True, color='green' if color else 'white',
                                      alpha=0.2))
                if text:
                    axs.text(m.midpoint[0], m.midpoint[1], '{}'.format(len(m.supported_b_splines)), ha='center',
                             va='center')
        if title:
            plt.title('dim(S) = {}'.format(len(self.S)))

        if not axes:
            plt.axis('off')

        if filename:
            plt.savefig(filename)

        plt.show()

    def refine(self, beta: float, error_function: typing.Callable, refinement_strategy='minimal') -> None:
        """
        Refine the LR-mesh in order to introduce beta * dim(S) new degrees of freedom.
        The error function takes an element and returns the elemental error contribution.
        :param refinement_strategy: the refinement strategy used for splitting a single element.
        :param beta: growth parameter
        :param error_function: evaluates the error contribution from a given element
        :return: None
        """

        previous_dim = len(self.S)
        number_of_inserted_lines = 0
        while len(self.S) <= previous_dim * (1 + beta):
            element_to_refine = max(self.M, key=error_function)

            if refinement_strategy is 'minimal':
                m = self.get_minimal_span_meshline(element_to_refine, axis=number_of_inserted_lines % 2)
            elif refinement_strategy is 'full':
                m = self.get_full_span_meshline(element_to_refine, axis=number_of_inserted_lines % 2)
            else:
                raise NotImplemented('The requested refinement strategy is not implemented yet')
            self.insert_line(m)
            number_of_inserted_lines += 1

    def mesh_to_array(self, N=20):
        """
        Returns the set of meshlines as an array of size (len(self.meshlines), 2, N) for transformation and plotting purposes (IGA).

        :param N: Number of samples along each meshline
        :return: np.ndarray
        """
        M = len(self.meshlines)
        meshlines = np.zeros((M, 2, N))

        for i, meshline in enumerate(self.meshlines):
            line_x = np.linspace(meshline.constant_value, meshline.constant_value, N)
            line_y = np.linspace(meshline.start, meshline.stop, N)

            if meshline.axis == 0:
                meshlines[i] = (line_x, line_y)
            else:
                meshlines[i] = (line_y, line_x)

        return meshlines

    def peelable(self):
        """
        Returns true if the peeling algorithms terminates with
        :return:
        """

        S_LD = self.S
        M_LD = self.M

        elements_to_remove = []
        for e in M_LD:
            if not e.is_overloaded():
                elements_to_remove.append(e)
                for b in e.supported_b_splines:
                    try:
                        S_LD.remove(b)
                    except ValueError:
                        pass

        changed = True
        while changed:
            changed = False
            for e in M_LD:
                candidates = [b for b in e.supported_b_splines if b in S_LD]
                if len(candidates) == 1:
                    M_LD.remove(e)
                    S_LD.remove(candidates[0])
                    changed = True

        if len(S_LD) == 0:
            return True
        else:
            return False

    def _element_cache(self):

        cache = {}
        for k, e in enumerate(self.M):
            i0 = _find_knot_interval(e.u_min, self.global_knots_u, endpoint=False)
            i1 = _find_knot_interval(e.u_max, self.global_knots_u, endpoint=False)
            j0 = _find_knot_interval(e.v_min, self.global_knots_v, endpoint=False)
            j1 = _find_knot_interval(e.v_max, self.global_knots_v, endpoint=False)
            for i in range(i0, i1):
                for j in range(j0, j1):
                    cache[(i, j)] = e
        self.element_cache = cache

    def get_element_containing_point(self, u, v):

        if self.element_cache is None:
            self._element_cache()

        i0 = _find_knot_interval(u, self.global_knots_u, endpoint=False)
        j0 = _find_knot_interval(v, self.global_knots_v, endpoint=False)

        return self.element_cache[(i0, j0)]

    def edge_functions(self):
        """
        Returns the indices of all B-splines corresponding to an edge-degree-of-freedom.
        :return: np.ndarray
        """

        idx = []
        for i in range(len(self.S)):
            if self.S[i].is_edge_dof():
                idx.append(i)
        return np.array(idx, dtype=np.int)

    def update_global_indices(self):
        for i, b in enumerate(self.S):
            b.id = i
