# Data layout

Data are deliberately excluded from this repository. Obtain each dataset from its official source and place it under `data/` as configured in `configs/`.

For every split directory, the loader expects:

```
Split_Folder/
  img/<image file>
  labelcol/<mask file>
  <split>_text.csv
```

The CSV must contain an `Image` column and either a `Description` or `text` column. For QaTa-COV19, masks are named `mask_<Image>`; for MosMedData+, masks use the image name directly. The loader detects these conventions automatically.
