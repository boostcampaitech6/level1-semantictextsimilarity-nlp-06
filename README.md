# level1-semantictextsimilarity-nlp-06
level1-semantictextsimilarity-nlp-06 created by GitHub Classroom

## Team members

김동언_T6014

임은형_T6146

함문정_T6184

허재하_T6187

황인수_T6192

이건하_T6194


## Performance



| **Learning Rate** | **Batch Size** | **Optimizer** | **Scheduler** | **Loss Function**| **Preprocessing**|**Performance**|
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|$2.5e^{-5}$|64|AdamP|OneCycleLR|MSELoss|`Augmentation`|0.917|
|$2.5e^{-5}$|64|AdamP|OneCycleLR|MSELoss|`None`|0.913|
|$2.5e^{-5}$|64|AdamP|OneCycleLR|L1Loss|`None`|0.921|
|$2.5e^{-5}$|64|AdamP|OneCycleLR|L1Loss|`Hanspell`|0.915|
|$2.5e^{-5}$|64|AdamP|OneCycleLR|L1Loss|`Hanspell`, `Augmentation`|0.923|
|$2.5e^{-5}$|64|AdamP|OneCycleLR|L1Loss|`Augmentation`|0.916|
|$5e^{-6}$|64|AdamP|'None'|L1Loss|`Hanspell`, `Augmentation`|0.919|
