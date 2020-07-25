# anonymous-project

## workflow

1. vectorize meta input (week, gender, etc.)
2. sort related ct-scan by z-axis
3. mask lung ct-scan sequence with to bins
4. use autoencoder to encode lung-mask to lower dimension
5. concatenate meta input and encoded ct-scan bins (Nmeta + Nbins * Ndims)
6. train
