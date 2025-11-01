required_pkgs <- c(
  "readr",
  "dplyr"
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("[INFO] installing %s ...", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  } else {
    message(sprintf("[INFO] %s already installed, skipping.", pkg))
  }
}

invisible(lapply(required_pkgs, install_if_missing))

message("[INFO] package installation step completed.")
