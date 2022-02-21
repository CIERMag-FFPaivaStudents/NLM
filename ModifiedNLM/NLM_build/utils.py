import warnings
import functools
import numpy as np

class deprecate_kwarg:
    """Decorator ensuring backward compatibility when argument names are
    modified in a function definition.
    Parameters
    ----------
    kwarg_mapping: dict
        Mapping between the function's old argument names and the new
        ones.
    warning_msg: str
        Optional warning message. If None, a generic warning message
        is used.
    removed_version : str
        The package version in which the deprecated argument will be
        removed.
    """

    def __init__(self, kwarg_mapping, warning_msg=None, removed_version=None):
        self.kwarg_mapping = kwarg_mapping
        if warning_msg is None:
            self.warning_msg = ("`{old_arg}` is a deprecated argument name "
                                "for `{func_name}`. ")
            if removed_version is not None:
                self.warning_msg += (f'It will be removed in '
                                     f'version {removed_version}.')
            self.warning_msg += "Please use `{new_arg}` instead."
        else:
            self.warning_msg = warning_msg

    def __call__(self, func):
        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            for old_arg, new_arg in self.kwarg_mapping.items():
                if old_arg in kwargs:
                    #  warn that the function interface has changed:
                    warnings.warn(self.warning_msg.format(
                        old_arg=old_arg, func_name=func.__name__,
                        new_arg=new_arg), FutureWarning, stacklevel=2)
                    # Substitute new_arg to old_arg
                    kwargs[new_arg] = kwargs.pop(old_arg)

            # Call the function with the fixed arguments
            return func(*args, **kwargs)
        return fixed_func


class channel_as_last_axis():
    """Decorator for automatically making channels axis last for all arrays.
    This decorator reorders axes for compatibility with functions that only
    support channels along the last axis. After the function call is complete
    the channels axis is restored back to its original position.
    Parameters
    ----------
    channel_arg_positions : tuple of int, optional
        Positional arguments at the positions specified in this tuple are
        assumed to be multichannel arrays. The default is to assume only the
        first argument to the function is a multichannel array.
    channel_kwarg_names : tuple of str, optional
        A tuple containing the names of any keyword arguments corresponding to
        multichannel arrays.
    multichannel_output : bool, optional
        A boolean that should be True if the output of the function is not a
        multichannel array and False otherwise. This decorator does not
        currently support the general case of functions with multiple outputs
        where some or all are multichannel.
    """
    def __init__(self, channel_arg_positions=(0,), channel_kwarg_names=(),
                 multichannel_output=True):
        self.arg_positions = set(channel_arg_positions)
        self.kwarg_names = set(channel_kwarg_names)
        self.multichannel_output = multichannel_output

    def __call__(self, func):
        @functools.wraps(func)
        def fixed_func(*args, **kwargs):

            channel_axis = kwargs.get('channel_axis', None)

            if channel_axis is None:
                return func(*args, **kwargs)

            # TODO: convert scalars to a tuple in anticipation of eventually
            #       supporting a tuple of channel axes. Right now, only an
            #       integer or a single-element tuple is supported, though.
            if np.isscalar(channel_axis):
                channel_axis = (channel_axis,)
            if len(channel_axis) > 1:
                raise ValueError(
                    "only a single channel axis is currently suported")

            if channel_axis == (-1,) or channel_axis == -1:
                return func(*args, **kwargs)

            if self.arg_positions:
                new_args = []
                for pos, arg in enumerate(args):
                    if pos in self.arg_positions:
                        new_args.append(np.moveaxis(arg, channel_axis[0], -1))
                    else:
                        new_args.append(arg)
                new_args = tuple(new_args)
            else:
                new_args = args

            for name in self.kwarg_names:
                kwargs[name] = np.moveaxis(kwargs[name], channel_axis[0], -1)

            # now that we have moved the channels axis to the last position,
            # change the channel_axis argument to -1
            kwargs["channel_axis"] = -1

            # Call the function with the fixed arguments
            out = func(*new_args, **kwargs)
            if self.multichannel_output:
                out = np.moveaxis(out, -1, channel_axis[0])
            return out

        return fixed_func

class deprecate_multichannel_kwarg(deprecate_kwarg):
    """Decorator for deprecating multichannel keyword in favor of channel_axis.
    Parameters
    ----------
    removed_version : str
        The package version in which the deprecated argument will be
        removed.
    """

    def __init__(self, removed_version='1.0', multichannel_position=None):
        super().__init__(
            kwarg_mapping={'multichannel': 'channel_axis'},
            warning_msg=None,
            removed_version=removed_version)
        self.position = multichannel_position

    def __call__(self, func):
        @functools.wraps(func)
        def fixed_func(*args, **kwargs):

            if self.position is not None and len(args) > self.position:
                warning_msg = (
                    "Providing the `multichannel` argument positionally to "
                    "{func_name} is deprecated. Use the `channel_axis` kwarg "
                    "instead."
                )
                warnings.warn(warning_msg.format(func_name=func.__name__),
                              FutureWarning,
                              stacklevel=2)
                if 'channel_axis' in kwargs:
                    raise ValueError(
                        "Cannot provide both a `channel_axis` kwarg and a "
                        "positional `multichannel` value."
                    )
                else:
                    channel_axis = -1 if args[self.position] else None
                    kwargs['channel_axis'] = channel_axis

            if 'multichannel' in kwargs:
                #  warn that the function interface has changed:
                warnings.warn(self.warning_msg.format(
                    old_arg='multichannel', func_name=func.__name__,
                    new_arg='channel_axis'), FutureWarning, stacklevel=2)

                # multichannel = True -> last axis corresponds to channels
                convert = {True: -1, False: None}
                kwargs['channel_axis'] = convert[kwargs.pop('multichannel')]

            # Call the function with the fixed arguments
            return func(*args, **kwargs)
        return fixed_func

def convert_to_float(image, preserve_range):
    """Convert input image to float image with the appropriate range.
    Parameters
    ----------
    image : ndarray
        Input image.
    preserve_range : bool
        Determines if the range of the image should be kept or transformed
        using img_as_float. Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html
    Notes
    -----
    * Input images with `float32` data type are not upcast.
    Returns
    -------
    image : ndarray
        Transformed version of the input.
    """
    if image.dtype == np.float16:
        return image.astype(np.float32)
    if preserve_range:
        # Convert image to double only if it is not single or double
        # precision float
        if image.dtype.char not in 'df':
            image = image.astype(float)
    else:
        from ..util.dtype import img_as_float
        image = img_as_float(image)
    return image
