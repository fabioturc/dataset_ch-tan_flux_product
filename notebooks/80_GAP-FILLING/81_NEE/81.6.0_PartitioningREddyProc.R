# LOAD LIBRARIES----------------------------------------------------------------
library(tidyverse)
library(REddyProc)
library(caTools)
library(data.table)
library(viridis)
library(mlegp)

# LOAD DATA---------------------------------------------------------------------
file_fluxes_meteo <- "notebooks/80_GAP-FILLING/81_NEE/81.5.2_PartitioningSubsetForREddyProc.csv"
filedata <- read_csv(file_fluxes_meteo, show_col_types = FALSE)

# SETTINGS----------------------------------------------------------------------
output_path = 'notebooks/80_GAP-FILLING/81_NEE/'

# FUNCTION----------------------------------------------------------------------
XG_Partition <- function(data, date_start, date_end) {
  
  # record output
  logf <- file(paste0(output_path, "81.6.1_logfile_PartitioningNEE_XGf.txt"))
  sink(logf, type = "output", split = TRUE, append = TRUE)
  sink(logf, append=TRUE, type="message")
  
  # load data
  df_XG <- data %>% 
    select(
      TIMESTAMP_END, 
      USTAR,
      ppfd, 
      vpd, 
      sw_in,
      ta, 
      rh, 
      NEE_L3.3_CUT_50_QCF0_gfXGBoost_parcelA_all, 
      NEE_L3.3_CUT_50_QCF0_gfXGBoost_parcelA_certain,
      NEE_L3.3_CUT_50_QCF0_gfXGBoost_parcelB_all,
      NEE_L3.3_CUT_50_QCF0_gfXGBoost_parcelB_certain,
      NEE_L3.3_CUT_50_QCF0_gfXGBoost_footprint
  )
  
  cat("Data selected \n")
  
  
  # Check timestamp
  cat("Check timestamp \n")
  print(head(df_XG$TIMESTAMP_END))
  
  # Convert timestamp to posix
  df_XG$TIMESTAMP_END <- as.POSIXct(df_XG$TIMESTAMP_END, format="%Y-%m-%d %H:%M:%S", tz = "Etc/GMT-1")
  
  assign("df_XG", df_XG, envir = .GlobalEnv)
  
  # stop and check using browser() 
  # type c in the console
  # Q is quite 
  
  # Prepare EddyProc data
  EddyData.F <- df_XG[FALSE]
  EddyData.F <- EddyData.F %>% 
    mutate(
    TIMESTAMP = df_XG$TIMESTAMP_END,
    NEE_Amix = as.numeric(as.character(df_XG$NEE_L3.3_CUT_50_QCF0_gfXGBoost_parcelA_all)),
    NEE_Bmix = as.numeric(as.character(df_XG$NEE_L3.3_CUT_50_QCF0_gfXGBoost_parcelB_all)),
    NEE_Acert = as.numeric(as.character(df_XG$NEE_L3.3_CUT_50_QCF0_gfXGBoost_parcelA_certain)),
    NEE_Bcert = as.numeric(as.character(df_XG$NEE_L3.3_CUT_50_QCF0_gfXGBoost_parcelB_certain)),
    NEE_footprint = as.numeric(as.character(df_XG$NEE_L3.3_CUT_50_QCF0_gfXGBoost_footprint)),
    Ustar = as.numeric(as.character(df_XG$USTAR)),
    Rg = as.numeric(as.character(df_XG$sw_in)),
    Tair = as.numeric(as.character(df_XG$ta)),
    RH = as.numeric(as.character(df_XG$rh)),
    VPD = pmax(as.numeric(as.character(df_XG$vpd)), 0)
    ) %>% 
    filter(TIMESTAMP >= as.POSIXct(paste0(date_start, ' 00:30:00'))) %>%
    filter(TIMESTAMP <= as.POSIXct(paste0(date_end, ' 00:00:00'))) %>%
    mutate(TIMESTAMP_STRING = as.character(TIMESTAMP)) %>%
    mutate(TIMESTAMP_STRING = ifelse(nchar(TIMESTAMP_STRING) == 10, 
                                     paste0(TIMESTAMP_STRING, " 00:00:00"), 
                                     TIMESTAMP_STRING))
  
  cat("Input data check \n")
  print(str(EddyData.F))
  cat("Input data ready \n")
  assign("XGEddyData.F", EddyData.F, envir = .GlobalEnv)
  

  
  # REddyProc processing
  EddyProc.C <- sEddyProc$new('CH-TAN', EddyData.F,
                              c('NEE_Amix', 'NEE_Acert', 'NEE_Bmix', 'NEE_Bcert', 'NEE_footprint',
                                'Rg', 'Tair', 'Ustar', 'VPD'),
                              ColPOSIXTime = "TIMESTAMP")
  EddyProc.C$sSetLocationInfo(LatDeg = 47.480620, LongDeg = 8.911868, TimeZoneHour = 1)
  
  # Check data
  cat("Check REddyProc timestamp \n")
  print(str(str(EddyProc.C)))
  print(head(EddyProc.C$sDATA$sDateTime))
  print(head(EddyProc.C$sTEMP)) # Middle timestamp
  print(head(EddyProc.C$sDATA))
  
  cat("REddyProc ready \n")
  cat("Start MDS Gapfilling \n")
  
  
  
  # MDS Gap-filling
  EddyProc.C$sMDSGapFill('Tair', FillAll = FALSE)
  EddyProc.C$sMDSGapFill('Rg', FillAll = FALSE)
  EddyProc.C$sMDSGapFill('VPD', FillAll = FALSE)
  cat("Meteo gapfilled \n")
  
  EddyProc.C$sMDSGapFill('NEE_Amix', FillAll = TRUE)
  cat("NEE of parcel A (version including mixed contribution data) gapfilled \n")
  
  EddyProc.C$sMDSGapFill('NEE_Acert', FillAll = TRUE)
  cat("NEE of parcel A (version with gapfilled data where mixed contribution data) gapfilled \n")
  
  EddyProc.C$sMDSGapFill('NEE_Bmix', FillAll = TRUE)
  cat("NEE of parcel B (version including mixed contribution data) gapfilled \n")
  
  EddyProc.C$sMDSGapFill('NEE_Bcert', FillAll = TRUE)
  cat("NEE of parcel B (version with gapfilled data where mixed contribution data) gapfilled \n")

  EddyProc.C$sMDSGapFill('NEE_footprint', FillAll = TRUE)
  cat("NEE of full footprint gapfilled\n")
  
  # Export filled data
  FillEddyData.F <- EddyProc.C$sExportResults()
  FillEddyData.F$TIMESTAMP <- EddyData.F$TIMESTAMP_STRING

  assign("XGFillEddyData.F", FillEddyData.F, envir = .GlobalEnv)
  
  # Partitioning
  cat("Start partitioning \n")
  
  EddyProc.C$sMRFluxPartition(suffix = 'Amix')
  EddyProc.C$sGLFluxPartition(suffix = 'Amix')
  cat("NEE of parcel A (version including mixed contribution data) partitioned\n")
  
  EddyProc.C$sMRFluxPartition(suffix = 'Acert')
  EddyProc.C$sGLFluxPartition(suffix = 'Acert')
  cat("NEE of parcel A (version with gapfilled data where mixed contribution data) partitioned\n")
  
  EddyProc.C$sMRFluxPartition(suffix = 'Bmix')
  EddyProc.C$sGLFluxPartition(suffix = 'Bmix')
  cat("NEE of parcel B (version including mixed contribution data) partitioned\n")
  
  EddyProc.C$sMRFluxPartition(suffix = 'Bcert')
  EddyProc.C$sGLFluxPartition(suffix = 'Bcert')
  cat("NEE of parcel B (version with gapfilled data where mixed contribution data) partitioned\n")
  
  EddyProc.C$sMRFluxPartition(suffix = 'footprint')
  EddyProc.C$sGLFluxPartition(suffix = 'footprint')
  cat("NEE of full footprint partitioned\n")


  # Export partitioned data
  FillPartitionEddyData.F <- EddyProc.C$sExportResults()
  FillPartitionEddyData.F$TIMESTAMP <- EddyData.F$TIMESTAMP_STRING
  
  Partition_path <- paste0(output_path, "/Partition")
  dir.create(Partition_path, recursive = TRUE)
  write.csv(FillPartitionEddyData.F, file = paste0(Partition_path, "/NEE_XG-GAPF_PART_ReddyProc.csv"), row.names = FALSE)

  cat(paste0("Files saved to: ", Partition_path, "\n"))
  assign("XGFillPartitionEddyData.F", FillPartitionEddyData.F, envir = .GlobalEnv)

  # Save fingerprint plots
  EddyProc.C$sPlotFingerprint('GPP_footprint_f', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('GPP_DT_footprint', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('Reco_footprint', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('Reco_DT_footprint', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('GPP_Amix_f', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('GPP_DT_Amix', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('Reco_Amix', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('Reco_DT_Amix', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('GPP_Bmix_f', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('GPP_DT_Bmix', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('Reco_Bmix', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('Reco_DT_Bmix', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('GPP_Acert_f', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('GPP_DT_Acert', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('Reco_Acert', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('Reco_DT_Acert', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('GPP_Bcert_f', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('GPP_DT_Bcert', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('Reco_Bcert', Dir = Regular_path)
  EddyProc.C$sPlotFingerprint('Reco_DT_Bcert', Dir = Regular_path)
  
  cat("Regular Partitioning done \n")

}

# RUN FUNCTION------------------------------------------------------------------
XG_Partition(filedata, '2023-11-07', '2025-06-05')

