"""
Module with common Hydro utils. Contains:
    - Namespaced utility class that can be adapted for individual scores
    - Structures (as named tuples) that can be used to store inputs and datasets for
      common operations

FUTUREWORK: If this pattern works well. It can be adapted to most scores.
"""

import abc
import typing
import xarray as xr

# TODO:
# - error handling
# - port scores with current utility in separate impl files, i.e. nse_impl,
#   pbias_impl - manually overriding base classes
# - common make and undo logic
# - adapter to deal with extra fields
# - adapter to deal with components
# - tests


class HydroMetaScore(typing.NamedTuple):
    """
    Used to store the result from running a score, including any metadata.

    If
    """

    # Scoring
    score: xr.Dataset
    include_components: bool
    score_components: dict[str, xr.Dataset]

    # metadata used to preserve original type
    is_dataarray: bool

    # readonly method, so it can be part of  the object
    def retrieve_result_from_metascore(self):
        """
        Default score retrieval method - should work with most scores:

        score (always)
            1. if the score was originally a dataarray, convert it back to a
               dataarray.
            2. force the array name to ``*Util.score_name`` for simplicity
            3. Nothing needs to be done for a dataset

        components (include_components = True):
            1. check that score_components are not empty
            2. resolve datasets v.s. dataarrays
                a. if components are datasets:
                    - add a coordinate for each component named "components"
                    - merge the datasets together
                b. otherwise components are dataarrays (enforced):
                    - make them single array datasets with name = *Utils.score_name
                      (from the appropriate Utils function)
                    - repeat a.
                    - this is so that a. and b. have consistent structure. Since
                      both cases produce datasets anyway.
            3. ``*Utils.score_name`` also merged as one of the components

        return score or score with components depending on ``include_components``
        """
        if ret_score_components:
            ...
        else:
            ...
            # assert result and force score name

    def _retrieve_score(self):
        """
        retrieves score from metascore
        see ``retrieve_result_from_metascore`` for details
        """
        ...

    def _retrieve_score_components(self):
        """
        retrieves components from metascore
        see ``retrieve_result_from_metascore`` for details
        """
        ...


class HydroMetaInputs(typing.NamedTuple):
    """
    A higher order structure that contains common data that hydro scoring
    functions can use.

    It provides a common interface to do preliminary checks

    It consolidates any input `XarrayLike` into `xr.Dataset`s.

    This is done so that any mathematical operations are more predictable to
    trace (and programatically check). When allowing mixed types i.e. operations
    that compound datasets with dataarrays, we may have undesired behaviour.

    .. important ::

        Any inner function using HydroMetaInputs ``MUST`` only deal with
        datasets. Compatiblity with dataarrays is ``automatically`` handled
        using a common transform/undo implementation.

        see:
            - ``HydroUtils.make_metainputs_from_inputs``
            - ``HydroUtils.retrieve_score_from_metascore``

    Usage:
        .. code-block :: python

            NseMetaDataset(HydroMetaDataset):
                is_angular: bool

            NseUtils(HydroUtils, score_name="NSE")
                @classmethod
                def make_metadata(cls, ...) -> NseMetaDataset:
                    # returns mutable metadataset
                    cls.make_common_metadataset()

                    #

                    cls.common_checks()
                    # do checks specific to Nse
    .. note::

        This is initialised like `collections.namedtuple`, and hence inherits
        useful traits of tuples - in particular, it is immutable after creation.
        Subclassing to NamedTuple just allows for easier type hinting, compared
        to the `collections` API.
    """

    # Input datasets - these are
    fcst: xr.Dataset
    obs: xr.Dataset
    weights: xr.Dataset | None

    # metadata used to preserve original type
    is_dataarray: bool

    # Populated after gather_dimensions call
    gathered_dims: FlexibleDimensionTypes

    # not read-only so it has to be a classmethod or staticmethod
    @classmethod
    def make_metainputs(
        cls,
        *,
        fcst: xr.Dataset,
        obs: xr.Dataset,
        weights: xr.Dataset | None,
        reduce_dims: FlexibleDimensionTypes | None,
        preserve_dims: FlexibleDimensionTypes | None,
    ) -> HydroMetaInputs:
        """
        Performs checks on common inputs above. Note that this will give a
        incomplete (but mostly complete and checked)

        For specific scores, the tuple needs to be de-referenced and
        reconstructued into their specific namespace, along with any custom
        checks.

        Example:

            code-block :: python

                # For NSE, we have `is_angular` as a field
                class NseMetaDataset(
                    ...
                    is_angular: bool
                    ...

                    def make_metainputs(self, fcst, ..., is_angular) -> NseMetaDataset:
                        # performs common checks, stores metadata and gathers dimensions
                        hydro_mds = super(HydroMetaInputs, self).make_metainputs(fcst, ...)

                        # extra safety check for angular
                        assert isinstance(is_angular, bool)

                        # re-assign to NseMetaDataset explicitly
                        return NseMetaDataset(
                            fcst=hydro_mds.fcst,
                            ...,
                            is_angular=is_angular,
                        )
        """


class HydroUtils:
    """
    Utility class for most hydro scores.

    Contains useful checks and structure creation methods. These can be
    subclassed and adapted for specific scores.

    The goal for this class is to provide:
        - any object creation helper functions for creating the namedtuple
          structures i.e. ``HydroMetaScore`` and ``HydroMetaInputs``.
          - these are overridable.
        - "common" checks that should be done on all hydro scores.
        - associated error messages to those checks
    """

    ERROR_CORRUPT_SCORE_METADATA: str = """
    CRITICAL: The underlying data array type is CORRUPT.

    Either multiple keys were detected for the wrapped dataset or the metadata
    associated with the HydroMetaInput or HydroMetaScore has been mutated.

    This is likely a BUG and NOT EXPECTED.

    Please raise an issue on github.com/nci/scores citing this error.
    """

    ERROR_MIXED_XR_DATA_TYPES: str = """
    `fcst`, `obs` and `weights` MUST BE the same type `xr.Dataset` OR
    `xr.DataArray` EXCLUSIVELY; NOT a mix of the two types.
    """

    #: datasets require a name - used when promoting a dataarray without a name.
    #: NOTE: it is FAIRLY COMMON that a dataarray DOES NOT have a name.
    _DATAARRAY_TEMPORARY_NAME: str = "__NONAME"

    def __init_subclass__(cls, *, score_name: str):
        """
        Initializer to force a ``score_name`` kwarg - used for ERROR messages.
        """
        super().__init_subclass__()
        cls.score_name = score_name

    @classmethod
    def error_with_scorename(cls, *, msg: str, error_type: Exception):
        """
        This helper should be used to raise any errors for hydro scores

        .. note::

            The constructed error is returned rather than raised here, so that
            the traceback actually reflects where the error is raised

        Usage:
            .. code-block :: python

                raise error_with_scorename(cls.ERROR_UHOH, ValueError)
        """
        return error_type(f"{cls.scorename}: {msg}")

    @classmethod
    def warn_with_scorename(cls, *, msg: str, warn_type: Warning):
        """
        This helper should be used to raise any warnings for hydro scores.

        Usage:
            .. code-block :: python

                warnings.warn(warn_with_scorename(cls.WARN_BLAH, UserWarning))
        """
        return warn_type(f"{cls.scorename}: {msg}")

    @classmethod
    def check_all_same_type(cls, *xrlike):
        """
        Hydro scores currently do NOT support mixed types, e.g. fcst, obs are
        datasets but weights is a dataarray as the behaviour is uncertain.

        Checks that all ``XarrayLike`` inputs are of the same type, i.e. either
        ALL datasets OR (exclusive OR i.e. xor) ALL dataarrays.

        Raises:
            TypeError: If mixed types are provided or if input isn't XarrayLike.
        """
        assert len(xrlike) > 0

        xrlike_remove_none = [_x for _x in xrlike if _x is not None]

        if not all_same_xarraylike(xrlike_remove_none):
            raise error_with_scorename(cls.ERROR_MIXED_XR_DATA_TYPES)

    @classmethod
    def check_metascore_consistency(
        cls,
        metainput: HydroMetaInput,
        metascore: HydroMetaScore,
    ):
        """
        Checks whether the meta information has been preserved.

        It is unlikely that meta information is mutated without actually
        manually extracting the underlying reference from the internal tuples
        and creating a new tuple.

        This is an extra safety to guard against such incorrect usage - which
        can lead to corrupt and inaccurate outputs.

        This check is usually done at the end before returning the final score,
        after it has been computed.

        Raises:
            RuntimeError: critical failure if metadata is corrupt.
        """
        corrupt = metascore.is_dataarray != metainput.is_dataarray

        if metascore.is_dataarray:
            # check that only one key is present if returning a data array
            key_names: list[str] = list(metascore.score.data_vars.keys())
            corrupt = corrupt or len(key_names) != 1

        if corrupt:
            raise error_with_scorename(cls.ERROR_CORRUPT_SCORE_METADATA)
