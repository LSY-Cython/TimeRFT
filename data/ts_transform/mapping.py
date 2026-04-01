import math
from collections.abc import Callable, Sequence

from data.ts_transform.base import Transformation
from data.ts_transform.transforms import (
    AddObservedMask,
    AddTimeIndex,
    AddVariateIndex,
    AddSampleIndex,
    DummyValueImputation,
    ExtendMask,
    FlatPackCollection,
    FlatPackFields,
    GetPatchSize,
    ImputeTimeSeries,
    MaskedPrediction,
    EvalMaskedPrediction,
    PackFields,
    PatchCrop,
    FinetunePatchCrop,
    EvalCrop,
    Patchify,
    SampleDimension,
    SelectFields,
    SequencifyField,
    EvalPad,
)


def pretrain_transform_map(hparams, seq_fields) -> Callable[..., Transformation]:
    """
    Get a dictionary of Transforms, with a default Transform as defined:
    SampleDimension: Subsample the variate dimension of a time series
    GetPatchSize: Get patch size for a given time series
    PatchCrop: Perform cropping on the time series
    PackFields: Pack each feature columns, including 'target' and 'past_feat_dynamic_real'.
    AddObservedMask: Add the observed_mask feature
    ImputeTimeSeries: Imputes missing values with 0
    Patchify: Perform patching
    AddVariateIndex: Add variate_id feature
    AddTimeIndex: Add time_id feature
    MaskedPrediction: Specify the task,
        i.e., sample the total input length, as well as sample the proportion of look-back window and prediction window length.
    ExtendMask: Add an auxiliary mask.
    FlatPackCollection: Pack/Merge along 'variate_id, time_id, prediction_mask, observed_mask, and target' dimensions.
    FlatPackFields: Pack/Merge 'target'.
    SequencifyField: sequencify the 'patch_size' field.
    SelectFields: Output the data of predefined fields

    :return: defaultdict with default Transform
    """

    def default_pretrain_transform():
        return (
            SampleDimension(
                max_dim=hparams["max_dim"],
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )
            + GetPatchSize(
                min_time_patches=hparams["min_patches"],
                target_field="target",
                patch_sizes=hparams["patch_sizes"],
            )
            + PatchCrop(
                min_time_patches=hparams["min_patches"],
                max_patches=hparams["max_seq_len"],
                will_flatten=True,
                offset=True,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )
            + PackFields(
                output_field="target",
                fields=("target",),
                feat=False,
            )
            + PackFields(
                output_field="past_feat_dynamic_real",
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
                feat=False,
            )
            + AddObservedMask(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                observed_mask_field="observed_mask",
                collection_type=dict,
            )
            + ImputeTimeSeries(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                imputation_method=DummyValueImputation(value=0.0),
            )
            + Patchify(
                max_patch_size=max(hparams["patch_sizes"]),
                fields=("target", "observed_mask"),
                optional_fields=("past_feat_dynamic_real",),
            )
            + AddVariateIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                variate_id_field="variate_id",
                expected_ndim=3,
                max_dim=hparams["max_dim"],
                randomize=True,
                collection_type=dict,
            )
            + AddTimeIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                time_id_field="time_id",
                expected_ndim=3,
                collection_type=dict,
            )
            + MaskedPrediction(
                min_mask_ratio=hparams["min_mask_ratio"],
                max_mask_ratio=hparams["max_mask_ratio"],
                target_field="target",
                truncate_fields=("variate_id", "time_id", "observed_mask"),
                optional_truncate_fields=("past_feat_dynamic_real",),
                prediction_mask_field="prediction_mask",
                expected_ndim=3,
            )
            + ExtendMask(
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
                mask_field="prediction_mask",
                expected_ndim=3,
            )
            + FlatPackCollection(
                field="variate_id",
                feat=False,
            )
            + FlatPackCollection(
                field="time_id",
                feat=False,
            )
            + FlatPackCollection(
                field="prediction_mask",
                feat=False,
            )
            + FlatPackCollection(
                field="observed_mask",
                feat=True,
            )
            + FlatPackFields(
                output_field="target",
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                feat=True,
            )
            + SequencifyField(field="patch_size", target_field="target")
            + SelectFields(fields=list(seq_fields))
        )

    return default_pretrain_transform()


def finetune_transform_map(
    hparams,
    offset: int,
    distance: int,
    prediction_length: int,
    context_length: int,
    patch_size: int,
    seq_fields: tuple
) -> Callable[..., Transformation]:

    def default_finetune_transform():
        return (
            GetPatchSize(
                min_time_patches=hparams["min_patches"],
                target_field="target",
                patch_sizes=hparams["patch_sizes"],
            )
            + FinetunePatchCrop(
                offset=offset,
                distance=distance,
                prediction_length=prediction_length,
                context_length=context_length,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )
            + PackFields(
                output_field="target",
                fields=("target",),
            )
            + PackFields(
                output_field="past_feat_dynamic_real",
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
            )
            + EvalPad(
                prediction_pad=-prediction_length % patch_size,
                context_pad=-context_length % patch_size,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )
            + AddObservedMask(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                observed_mask_field="observed_mask",
                collection_type=dict,
            )
            + ImputeTimeSeries(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                imputation_method=DummyValueImputation(value=0.0),
            )
            + Patchify(
                max_patch_size=max(hparams["patch_sizes"]),
                fields=("target", "observed_mask"),
                optional_fields=("past_feat_dynamic_real",),
            )
            + AddVariateIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                variate_id_field="variate_id",
                expected_ndim=3,
                max_dim=hparams["max_dim"],
                randomize=False,  # Disable random variate id
                collection_type=dict,
            )
            + AddTimeIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                time_id_field="time_id",
                expected_ndim=3,
                collection_type=dict,
            )
            + AddSampleIndex(  # Since sequence packing is not used, need to add sample id in transformation
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                sample_id_field="sample_id",
                expected_ndim=3,
                collection_type=dict,
            )
            + EvalMaskedPrediction(
                mask_length=math.ceil(prediction_length / patch_size),
                target_field="target",
                truncate_fields=(
                    "variate_id",
                    "time_id",
                    "observed_mask",
                    "sample_id",
                ),
                optional_truncate_fields=("past_feat_dynamic_real",),
                prediction_mask_field="prediction_mask",
                expected_ndim=3,
            )
            + ExtendMask(
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
                mask_field="prediction_mask",
                expected_ndim=3,
            )
            + FlatPackCollection(
                field="variate_id",
                feat=False,
            )
            + FlatPackCollection(
                field="time_id",
                feat=False,
            )
            + FlatPackCollection(
                field="sample_id",
                feat=False,
            )
            + FlatPackCollection(
                field="prediction_mask",
                feat=False,
            )
            + FlatPackCollection(
                field="observed_mask",
                feat=True,
            )
            + FlatPackFields(
                output_field="target",
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                feat=True,
            )
            + SequencifyField(field="patch_size", target_field="target")
            + SelectFields(fields=list(seq_fields))
        )

    return default_finetune_transform()


def val_transform_map(
    hparams: dict,
    offset: int,
    distance: int,
    prediction_length: int,
    context_length: int,
    patch_size: int,
    seq_fields: tuple
) -> Callable[..., Transformation]:

    def default_val_transform():
        return (
            GetPatchSize(
                min_time_patches=2,
                target_field="target",
                patch_sizes=hparams["patch_sizes"],
            )
            + EvalCrop(
                offset,
                distance,
                prediction_length,
                context_length,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )
            + PackFields(
                output_field="target",
                fields=("target",),
            )
            + PackFields(
                output_field="past_feat_dynamic_real",
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
            )
            + EvalPad(
                prediction_pad=-prediction_length % patch_size,
                context_pad=-context_length % patch_size,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )
            + AddObservedMask(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                observed_mask_field="observed_mask",
                collection_type=dict,
            )
            + ImputeTimeSeries(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                imputation_method=DummyValueImputation(value=0.0),
            )
            + Patchify(
                max_patch_size=max(hparams["patch_sizes"]),
                fields=("target", "observed_mask"),
                optional_fields=("past_feat_dynamic_real",),
            )
            + AddVariateIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                variate_id_field="variate_id",
                expected_ndim=3,
                max_dim=hparams["max_dim"],
                randomize=False,
                collection_type=dict,
            )
            + AddTimeIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                time_id_field="time_id",
                expected_ndim=3,
                collection_type=dict,
            )
            + AddSampleIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                sample_id_field="sample_id",
                expected_ndim=3,
                collection_type=dict,
            )
            + EvalMaskedPrediction(
                mask_length=math.ceil(prediction_length / patch_size),
                target_field="target",
                truncate_fields=(
                    "variate_id",
                    "time_id",
                    "observed_mask",
                    "sample_id",
                ),
                optional_truncate_fields=("past_feat_dynamic_real",),
                prediction_mask_field="prediction_mask",
                expected_ndim=3,
            )
            + ExtendMask(
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
                mask_field="prediction_mask",
                expected_ndim=3,
            )
            + FlatPackCollection(
                field="variate_id",
                feat=False,
            )
            + FlatPackCollection(
                field="time_id",
                feat=False,
            )
            + FlatPackCollection(
                field="sample_id",
                feat=False,
            )
            + FlatPackCollection(
                field="prediction_mask",
                feat=False,
            )
            + FlatPackCollection(
                field="observed_mask",
                feat=True,
            )
            + FlatPackFields(
                output_field="target",
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                feat=True,
            )
            + SequencifyField(field="patch_size", target_field="target")
            + SelectFields(fields=list(seq_fields))
        )

    return default_val_transform()
