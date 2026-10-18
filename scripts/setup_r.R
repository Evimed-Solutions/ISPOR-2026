# Install the exact diagnostic packages used for this review package.
# Run: Rscript --vanilla scripts/setup_r.R
repo <- "https://cloud.r-project.org"
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes", repos = repo)
if (!requireNamespace("cluster", quietly = TRUE) || as.character(packageVersion("cluster")) != "2.1.8.1") {
  remotes::install_version("cluster", version = "2.1.8.1", repos = repo, upgrade = "never")
}
remotes::install_github("Pechli-Lab/SurvdigitizeR@b9d2ad9500311189ba8adc2ec57831ad8b6042d5",
                       upgrade = "never", dependencies = NA)
