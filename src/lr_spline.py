import typing
from typing import List

import numpy as np

from src.aux_split_functions import split_single_basis_function
from src.meshline import Meshline
from src.b_spline import BSpline
from src.element import Element

Vector = typing.Union[typing.List['float'], np.ndarray]


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
            basis.append(BSpline(d1, d2, ku[i: i + d1 + 2], kv[j: j + d2 + 2]))

    for b in basis:
        for e in elements:
            if b.add_to_support_if_intersects(e):
                e.add_supported_b_spline(b)

    for i in range(len(unique_ku)):
        new_m = Meshline(start=unique_kv[0], stop=unique_kv[-1], constant_value=unique_ku[i], axis=0)
        new_m.set_multiplicity(ku)
        meshlines.append(new_m)
    for i in range(len(unique_kv)):
        new_m = Meshline(start=unique_ku[0], stop=unique_ku[-1], constant_value=unique_kv[i], axis=1)
        new_m.set_multiplicity(kv)
        meshlines.append(new_m)

    return LRSpline(elements, basis, meshlines)


class LRSpline(object):
    """
    Represents a LRSpline, which is a tuple (M, S), where M is a mesh and S is a set of basis functions
    defined on M.
    """

    def __init__(self, mesh: List['Element'], basis: List['BSpline'], meshlines: List['Meshline']) -> None:
        """
        Initialize an LR Spline with associated set of elements, and set of basis functions.
        :param mesh: elements constituting the LR mesh
        :param basis: basis functions defined over the LR mesh
        """
        self.M = mesh
        self.S = basis
        self.meshlines = meshlines

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
        least one BSpline. :param e: element to refine
        """
        raise NotImplementedError('LRSpline.{} is not implemented yet'.format(self.refine_by_element_minimal.__name__))

    @staticmethod
    def get_minimal_span_meshline(e: Element, axis=0) -> Meshline:
        """
        Finds the shortest possible meshline in direction prescribed by axis that splits at least one supported
        B-spline on the element. :param e: element to refine by :param axis: direction to look for split, 0 vertical,
        1 horizontal :return: minimal span meshline
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

        return Meshline(smallest_start, smallest_stop, constant_value, axis)

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
        meshline_already_exists = self.merge_meshlines(meshline)

        if meshline_already_exists:
            return

        # step 1
        # split B-splines against new meshline

        new_functions = []
        functions_to_remove = []
        for basis in self.S:
            # debug BSpline(1, 1, [3, 3.5, 4], [2, 3, 4])
            if meshline.splits_basis(basis):
                if meshline.number_of_knots_contained(basis) < meshline.multiplicity:
                    self.local_split(basis, meshline, functions_to_remove, new_functions)

        purged_S = [s for s in self.S if s not in functions_to_remove]

        self.S = purged_S
        # step 2
        # split new B-splines against old meshlines

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
        self.meshlines.append(meshline)

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

    def local_split(self, basis, m, functions_to_remove, new_functions):
        b1, b2, a1, a2 = split_single_basis_function(m, basis, return_weights=True)
        if self.contains_basis_function(b1):
            self._update_old_basis_function(basis, b1, a1, self.S)
        elif b1 in new_functions:
            i = new_functions.index(b1)
            b1_old = new_functions[i]
            b1_old.coefficient = b1_old.weight * b1_old.coefficient + b1.weight * b1.coefficient
            b1_old.weight = b1_old.weight + b1.weight
            new_functions[i] = b1_old
        else:
            b1.coefficient = basis.coefficient
            b1.weight = a1 * basis.weight
            new_functions.append(b1)
        if self.contains_basis_function(b2):
            self._update_old_basis_function(basis, b2, a2, self.S)
        elif b2 in new_functions:

            i = new_functions.index(b2)
            b2_old = new_functions[i]
            b2_old.coefficient = b2_old.weight * b2_old.coefficient + b2.weight * b2.coefficient
            b2_old.weight = b2_old.weight + b2.weight
            new_functions[i] = b2_old
        else:
            b2.coefficient = basis.coefficient
            b2.weight = a2 * basis.weight
            new_functions.append(b2)
        functions_to_remove.append(basis)

    @staticmethod
    def _update_old_basis_function(original_basis, new_basis, weight, basis_list) -> None:
        """
        Updates the basis function corresponding to b1 with new weights and coefficients, dependent on
        the basis that was split, and the new basis function.
        :param original_basis: the orginal basis function that was split, yielding `new basis`
        :param new_basis: the `new basis` originating from splitting `original_basis`, which is already present in self.S
        :return:
        """
        i = basis_list.index(new_basis)
        basis_list[i].coefficient = (basis_list[
                                         i].coefficient * new_basis.weight + original_basis.coefficient * original_basis.weight * weight) / (
                                            new_basis.weight + weight * original_basis.weight)
        basis_list[i].weight = new_basis.weight + weight * original_basis.weight

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

        e = self._find_element_containing_point(u, v)

        total = 0
        for b in e.supported_b_splines:
            total += b.coefficient * b(u, v)
        return total

    def _find_element_containing_point(self, u, v):
        for e in self.M:
            if e.contains(u, v):
                break
        else:
            raise ValueError('({}, {}) is not in the domain'.format(u, v))
        return e

    def merge_meshlines(self, meshline: Meshline) -> bool:
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
        meshlines_to_remove = []        for old_meshline in self.meshlines:
            if not old_meshline._similar(meshline):
                # the two meshlines are not comparable, continue.
                continue
            if meshline == old_meshline:
                # meshline already exists, no point in continuing
                return True
            if old_meshline.contains(meshline):
                # meshline is completely contained in the old meshline

                if meshline.multiplicity > old_meshline.multiplicity:
                    if abs(old_meshline.start - meshline.start) < tol and abs(old_meshline.stop - meshline.stop) < tol:
                        # if the new multiplicity is greater than the old one, and the endpoints coincide, keep the
                        # new line and remove the old line.
                        meshlines_to_remove.append(old_meshline)

                elif old_meshline.multiplicity >= meshline.multiplicity:
                    return True

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
        self.meshlines.append(meshline)
        return False
