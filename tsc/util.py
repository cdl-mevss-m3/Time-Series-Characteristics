import collections

import pandas as pd
from tqdm.auto import tqdm


def get_feature_combination_df(
        feature_combination: list,
        df: pd.DataFrame,
        index_levels: list,
        column_value: str = "value",
        verbose: bool = False
):
    """
    For each feature combination in ``feature_combinations``, a DataFrame is created that holds the
    calculated features and a list of these DataFrames is then returned.

    :param feature_combination: A feature combination is a list of tuples where the first entry is the
        function object (this function must have one parameter in the first position that represents
        the data for which to calculate the feature) and the second entry is the dictionary containing
        the function's parameters (can be empty or None if the function does not have any parameters
        other than the required positional one). For convenience, it is also allowed to directly specify
        the function object instead of the tuple with the function object and parameters. A DataFrame
        holding the calculated features will be created and then returned.
    :param df: The DataFrame for which to calculate the different features. The DataFrame must have
        a multi-index where accessing an element using one of the IDs in ``index_levels`` yields another
        DataFrame that can then be accessed via ``column_value`` to finally return a Series of values.
    :param index_levels: The index levels specifying the unique IDs. The features are calculated for each
        unique identifier (ID), where an identifier is a unique tuple with entries from this list. The
        order of this list must match the order in the index of ``df``.
    :param column_value: The value column of ``df`` (default: "value").
    :param verbose: If True, show a progress bar for each identifier in ``df`` (default: False).
    :return: A DataFrames that contains the calculated features.
    """
    # get the set of unique keys (index tuples)
    keys = set(zip(*[df.index.get_level_values(i) for i in index_levels]))
    
    cols = dict()
    for key in tqdm(keys, "IDs", disable=not verbose):
        data = df.loc[key, :][column_value]
        
        for func_params in feature_combination:
            # just for convenience so the user can write feature combinations with functions that have
            # no parameters more easily: [(my_func, None), (other_func, None)] vs [my_func, other_func]
            if callable(func_params):
                func_params = (func_params, dict())
            func, params = func_params
            
            if not params:
                func_result = func(data)
                _process_func_result(func, params, func_result, cols)
            else:
                func_result = func(data, **params)
                _process_func_result(func, params, func_result, cols)
    
    return pd.DataFrame(cols, index=pd.MultiIndex.from_tuples(keys, names=index_levels))


def _process_func_result(func, params, func_result, cols):
    params_str = create_params_str(params)
    
    if isinstance(func_result, collections.Iterable):
        for key, val in transform_func_result(func_result):
            column = create_func_result_identifier(func, params_str, key)
            cols[column] = cols.get(column, []) + [val]
    else:
        # primitive type
        column = create_func_result_identifier(func, params_str)
        cols[column] = cols.get(column, []) + [func_result]


def copy_doc_from(obj, append=True):
    """
    Decorator that copies the ``__doc__`` string from the specified object ``obj`` to the
    decorated function. If the decorated function already has a documentation attached,
    ``append`` determines whether to append the copied documentation to the existing one or
    to drop the copied documentation.

    :param obj: The object whose ``__doc__`` string should be copied to the decorated function.
    :param append: If True, append the copied documentation to the existing documentation.
        This parameter is ignored if the decorated function does not have any documentation
        attached (default: True).
    :return: The decorated function with the copied (and possibly appended) documentation.
    """
    
    def decorator(func):
        copied_doc = obj.__doc__ if hasattr(obj, "__doc__") else None
        if copied_doc:
            if func.__doc__ and append:
                func.__doc__ = func.__doc__ + "\n" + copied_doc
            else:
                func.__doc__ = copied_doc
        return func
    
    return decorator


def get_overridden_methods(cls):
    """
    Returns a set of all overridden methods of the class ``cls``.
    """
    # collect all attributes inherited from parent classes
    parent_attrs = set()
    for base in cls.__bases__:
        parent_attrs.update(dir(base))
    
    # find all methods implemented in the class itself
    methods = {name for name, thing in vars(cls).items() if callable(thing)}
    
    # return the intersection of both
    return parent_attrs.intersection(methods)


def transform_func_result(func_result: collections.Iterable):
    """
    Transforms the function result (any collections.Iterable) into an object which supports
    tuple-iteration like:

    >>> for key, value in transform_func_result(func_result):

    **IMPORTANT NOTE:**
        This function is rather specific to time series characteristic features (such as tsfresh),
        so it should be used only internally.
    """
    # case 1) tsfresh with zip return: nothing to do, already supports tuple-iteration
    if isinstance(func_result, collections.Iterator):
        pass
    # case 2) tsfresh with list/tuple return containing lists/tuples of size 2: also nothing to do, already supports tuple-iteration
    elif isinstance(func_result, (list, tuple)) and isinstance(func_result[0], (list, tuple)) and len(func_result[0]) == 2:
        pass
    # case 3) pd.Series or dict return: just use ".items()" instead to support tuple-iteration
    elif isinstance(func_result, (pd.Series, dict)):
        func_result = func_result.items()
    # case 4) remaining iterable return types: just name the entries 0 .. len(entries) - 1
    else:
        func_result = enumerate(func_result)
    return func_result


def create_func_result_identifier(func, params_str, key=None, key_separator="__"):
    """
    Creates a string of the following format:

    If ``key`` is None:

    ``<FUNC_NAME><PARAMS_STR>``

    If ``key`` is not None:

    ``<FUNC_NAME><PARAMS_STR>__key``

    In both cases, ``<FUNC_NAME>`` represents the name of the function object ``func``
    (via ``func.__name__``) and ``<PARAMS_STR>`` represents a string object given by
    ``params_str`` (e.g., obtained via the method ``create_params_str``).
    In the latter case, ``key`` represents the function return identifier (e.g., for
    a multi-value result, the ``key`` is the identifier for one such value) and it
    is separated by a double underscore.

    The double underscore separator can be changed by specifying the default
    parameters ``key_separator``.

    **IMPORTANT NOTE:**
        This function is rather specific to time series characteristic features (such as tsfresh),
        so it should be used only internally.
    """
    return f"{func.__name__}{params_str}{key_separator}{key}" if key is not None else f"{func.__name__}{params_str}"


def create_params_str(params, key_value_separator="_", param_separator="__"):
    """
    All ``(key, value)`` pairs/parameters in ``params`` are concatenated using a single underscore
    to separate ``key`` and ``value`` and a double underscore is prepended to each parameter
    to separate the parameters, i.e., a string in the following format is created

    ``__key1_value1__key2_value2 ... __keyN_valueN``

    If ``params`` is None or empty, an empty string is returned.

    The single underscore and double underscore separators can be changed by specifying
    the default parameters ``key_value_separator`` and ``param_separator``.

    **IMPORTANT NOTE:**
        This function is rather specific to time series characteristic features (such as tsfresh),
        so it should be used only internally.
    """
    return "".join([f"{param_separator}{k}{key_value_separator}{v}" for k, v in params.items()]) if params else ""
