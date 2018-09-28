from collections import defaultdict


def overloads_per_level(LR):
    """
    Returns a dictionary containing the number of overloaded elements per level, as well as the corresponding
    percentage of overloaded elements at each level.
    :return:
    """

    total_elements = len(LR.M)
    statistics = defaultdict(int)

    for element in LR.M:
        if element.is_overloaded():
            statistics[element.level] += 1
        else:
            statistics[element.level] += 0

    for level in statistics:
        statistics[level] = [statistics[level], statistics[level] / total_elements]

    return statistics
