# Re-runs SurvdigitizeR steps 1-4 exactly as the KM digitiser app does (run_survdigitizer.R:
# axes_identify and fig_clean with bg_lightness = 0.3, then safe_color_cluster with k = 2 and
# enhance = FALSE) and dumps every pixel that reaches colour clustering.
#
# Outputs:
#   work/pixels/<image_id>_pixels.csv   one row per clustered pixel: array row/col, h, s, l, cluster
#   results/diagnostics/cluster_summary.csv         per image: pixel count and the two cluster medoids
#
# Run: Rscript dump_clusters.R   (needs SurvdigitizeR and cluster)

root <- Sys.getenv("ISPOR_RUN_ROOT")
if (!nzchar(root)) stop("Run this stage using: python run.py figures")
in_dir <- file.path(root, "intermediates", "in")
px_dir <- file.path(root, "intermediates", "pixels")
out_dir <- file.path(root, "results", "diagnostics")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

dir.create(in_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(px_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(SurvdigitizeR))
ids <- read.csv(file.path(root, "data", "truth", "image_manifest.csv"))$image_id

rows <- list()
for (id in ids) {
  # img_read chooses the JPEG reader if the path contains "jpeg", so copy to a neutral name,
  # as the app does with its uploads.
  safe <- file.path(in_dir, paste0(gsub("jpeg", "jq", id), ".png"))
  file.copy(file.path(root, "data", "images", paste0(id, ".png")), safe, overwrite = TRUE)

  step1 <- SurvdigitizeR:::img_read(path = safe)
  step2 <- SurvdigitizeR:::axes_identify(fig.hsl = step1, bg_lightness = 0.3)
  step3 <- SurvdigitizeR:::fig_clean(fig.hsl = step2$fig.hsl, bg_lightness = 0.3,
                                      attempt_OCR = FALSE, word_sensitivity = 30)
  set.seed(260619)  # stable diagnostic sampling; primary observed extraction remains unchanged
  fit <- cluster::clara(x = step3[, c("h", "s", "l")], sampsize = min(500, max(1, nrow(step3))),
                        k = 2, samples = 50, pamLike = TRUE)

  # Crop indices map back through the axes; the array is stored bottom-up, so
  # row from the top of the image = image height - arr_row.
  write.csv(data.frame(arr_row = step2$axes$yaxis[step3$y], arr_col = step2$axes$xaxis[step3$x],
                       h = step3$h, s = step3$s, l = step3$l, group = fit$clustering),
            file.path(px_dir, paste0(id, "_pixels.csv")), row.names = FALSE)

  med <- fit$medoids
  rows[[id]] <- data.frame(
    image_id = id, img_rows = dim(step1)[1], img_cols = dim(step1)[2], n_pixels = nrow(step3),
    cluster1_hue = med[1, "h"], cluster1_saturation = med[1, "s"], cluster1_lightness = med[1, "l"],
    cluster1_size = fit$clusinfo[1, "size"],
    cluster2_hue = med[2, "h"], cluster2_saturation = med[2, "s"], cluster2_lightness = med[2, "l"],
    cluster2_size = fit$clusinfo[2, "size"]
  )
  cat(id, nrow(step3), "pixels\n")
}
write.csv(do.call(rbind, rows), file.path(out_dir, "cluster_summary.csv"), row.names = FALSE)
cat("wrote results/diagnostics/cluster_summary.csv\n")
