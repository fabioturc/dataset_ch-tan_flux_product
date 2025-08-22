
# Basic settings ----------------------------------------------------------

library(REddyProc)
library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)


# Set time info and output folders ----------------------------------------

# RUN ID
run_id <- format(Sys.time(), "%Y%m%d",tz="Etc/GMT-1")   ## To give unique ID to saved output files

Sys.setenv(TZ = "Etc/GMT-1")

set.seed(40)

# Read data ---------------------------------------------------------------

data <- read.csv('./50_MERGE_DATA_FLUXES/51.2_FluxProcessingChain_L3.3_subset-forREddyProcGapFilling.csv')

colnames(data)

df <- data %>%
  select(TIMESTAMP_END,
         VPD,
         SW_IN,
         TA,
         NEE_L3.1_L3.3_CUT_16_QCF,
         NEE_L3.1_L3.3_CUT_50_QCF,
         NEE_L3.1_L3.3_CUT_84_QCF,
         LE_L3.1_L3.3_CUT_NONE_QCF,
         H_L3.1_L3.3_CUT_NONE_QCF
  )

# Set TIMESTAMP -----------------------------------------------------------

head(df$TIMESTAMP_END)
str(df)

df$TIMESTAMP_END <- as.POSIXct(df$TIMESTAMP_END, format="%Y-%m-%d %H:%M:%S", tz = "Etc/GMT-1")

head(df$TIMESTAMP_END)
str(df)
summary(df)


# Build data for REddyProc ------------------------------------------------

EddyData.F <- df[FALSE]
EddyData.F$TIMESTAMP <- as.POSIXct(df$TIMESTAMP_END, format = '%Y-%m-%d %H:%M:%S', tz = Sys.timezone())

head(EddyData.F$TIMESTAMP)

# NEE
EddyData.F$NEE_U16 <- as.numeric(as.character(df$NEE_L3.1_L3.3_CUT_16_QCF))
EddyData.F$NEE_U50 <- as.numeric(as.character(df$NEE_L3.1_L3.3_CUT_50_QCF))
EddyData.F$NEE_U84 <- as.numeric(as.character(df$NEE_L3.1_L3.3_CUT_84_QCF))

EddyData.F$LE <- as.numeric(as.character(df$LE_L3.1_L3.3_CUT_NONE_QCF))
EddyData.F$H <- as.numeric(as.character(df$H_L3.1_L3.3_CUT_NONE_QCF))

# meteo
EddyData.F$Rg <- as.numeric(as.character(df$SW_IN))
# EddyData.F$Rg[EddyData.F$Rg < -20] <- NA  # To remove -9999 missing values
# EddyData.F$Rg[EddyData.F$Rg < 0] <- 0  # Below zero not accepted by ReddyProc
# EddyData.F$Rg[EddyData.F$Rg > 1200] <- 1200  # Above 1200 not accepted by ReddyProc

EddyData.F$Tair <- as.numeric(as.character(df$TA))
# EddyData.F$Tair[EddyData.F$Tair < -20] <- NA  # To remove -9999 missing values

EddyData.F$VPD <- as.numeric(as.character(df$VPD))

# cut data for 2021-2023
EddyData.F <- subset (EddyData.F, TIMESTAMP >= as.POSIXct('2021-01-01 00:30:00'))  # Date with first fluxes
EddyData.F <- subset (EddyData.F, TIMESTAMP <= as.POSIXct('2024-01-01 00:00:00'))


summary(EddyData.F)

# REddyProc ---------------------------------------------------------------
EddyProc.C <-sEddyProc$new('CH-OE2',  EddyData.F,
                           c('NEE_U16', 'NEE_U50', 'NEE_U84','LE', 'H', 'Rg','Tair', 'VPD'), ColPOSIXTime = "TIMESTAMP")   
EddyProc.C$sSetLocationInfo(LatDeg = 47.286417, LongDeg = 7.733750, TimeZoneHour = 1) # coord OE2
str(EddyProc.C)
head(EddyProc.C$sDATA$sDateTime)
head(EddyProc.C$sTEMP)
head(EddyProc.C$sDATA)


# MDS ---------------------------------------------------------------------
EddyProc.C$sMDSGapFill('Tair', FillAll = FALSE)
EddyProc.C$sMDSGapFill('Rg', FillAll = FALSE)
EddyProc.C$sMDSGapFill('VPD', FillAll = FALSE)


EddyProc.C$sMDSGapFill(Var = 'NEE_U16', FillAll = TRUE)
EddyProc.C$sMDSGapFill(Var = 'NEE_U50', FillAll = TRUE)
EddyProc.C$sMDSGapFill(Var = 'NEE_U84', FillAll = TRUE)

EddyProc.C$sMDSGapFill('LE', FillAll = TRUE, isVerbose = TRUE)

EddyProc.C$sMDSGapFill('H', FillAll = TRUE, isVerbose = TRUE)

## Calculate ET from LE (QC=0)
EddyProc.C$sTEMP$ET_f <- fCalcETfromLE(EddyProc.C$sTEMP$LE_f, EddyProc.C$sTEMP$Tair_f)

summary(EddyProc.C$sTEMP)


EddyProc.C$sPlotFingerprintY('NEE_U16_orig', Year = 2021)
EddyProc.C$sPlotFingerprintY('NEE_U50_orig', Year = 2021)
EddyProc.C$sPlotFingerprintY('NEE_U84_orig', Year = 2021)

EddyProc.C$sPlotFingerprintY('NEE_U16_f', Year = 2021)
EddyProc.C$sPlotFingerprintY('NEE_U50_f', Year = 2021)
EddyProc.C$sPlotFingerprintY('NEE_U84_f', Year = 2021)

EddyProc.C$sPlotFingerprintY('NEE_U16_orig', Year = 2022)
EddyProc.C$sPlotFingerprintY('NEE_U50_orig', Year = 2022)
EddyProc.C$sPlotFingerprintY('NEE_U84_orig', Year = 2022)

EddyProc.C$sPlotFingerprintY('NEE_U16_f', Year = 2022)
EddyProc.C$sPlotFingerprintY('NEE_U50_f', Year = 2022)
EddyProc.C$sPlotFingerprintY('NEE_U84_f', Year = 2022)

EddyProc.C$sPlotFingerprintY('NEE_U16_orig', Year = 2023)
EddyProc.C$sPlotFingerprintY('NEE_U50_orig', Year = 2023)
EddyProc.C$sPlotFingerprintY('NEE_U84_orig', Year = 2023)

EddyProc.C$sPlotFingerprintY('NEE_U16_f', Year = 2023)
EddyProc.C$sPlotFingerprintY('NEE_U50_f', Year = 2023)
EddyProc.C$sPlotFingerprintY('NEE_U84_f', Year = 2023)

# COLLECT DATA 
FilledEddyData.F <- EddyProc.C$sExportResults()

colnames(FilledEddyData.F)
FilledEddyData.F$TIMESTAMP <- EddyData.F$TIMESTAMP
colnames(FilledEddyData.F)


# Partitioning ---------------------------------------------------------------

# PARTITIONING
EddyProc.C$sMRFluxPartition(suffix = 'U16') # night-time partitioning
#EddyProc.C$sGLFluxPartition(suffix = 'U16') # day-time partitioning needs a package not available in CRAN anymore


# PARTITIONING
EddyProc.C$sMRFluxPartition(suffix = 'U84') # night-time partitioning
#EddyProc.C$sGLFluxPartition(suffix = 'U84')


# PARTITIONING
EddyProc.C$sMRFluxPartition(suffix = 'U50') # night-time partitioning
#EddyProc.C$sGLFluxPartition(suffix = 'U50')

# Warning message:
#   In EddyProc.C$sGLFluxPartition(suffix = "U84") :
#   replacing existing output columnsFP_VARday, NEW_FP_Temp, NEW_FP_VPD, FP_RRef_Night, FP_qc, FP_dRecPar, FP_errorcode, FP_GPP2000, 
#   FP_OPT_VPD, FP_OPT_NoVPD, FP_k, FP_beta, FP_alpha, FP_RRef, FP_E0, FP_k_sd, FP_beta_sd, FP_alpha_sd, FP_RRef_sd, FP_E0_sd
# So run partitioning for U50 last



# Export data-------------------------------------------------------------------
df <- EddyProc.C$sExportResults()
colnames(df)
df$TIMESTAMP_END = EddyData.F$TIMESTAMP

# define the flux columns we want to keep
# Using NEE partitioned fluxes with the night-time method 
cols = c('TIMESTAMP_END', 
         'NEE_U16_f', 'Reco_U16', 'GPP_U16_f',
         'NEE_U50_f', 'Reco_U50', 'GPP_U50_f',
         'NEE_U84_f', 'Reco_U84', 'GPP_U84_f',
         'LE_f', 'H_f', 'ET_f') 

gf_part_df = df %>% dplyr::select(all_of(cols))

# Plot data---------------------------------------------------------------------
indat = gf_part_df
plot_flux = function(indat, start, end, var){
  indat_zoom = indat %>% 
    dplyr::filter(TIMESTAMP_END > as.POSIXct(start) & TIMESTAMP_END < as.POSIXct(end))
  
  ggplot() +
    geom_line(data = indat_zoom,
              aes(x = TIMESTAMP_END, y = get(var))) +
    xlab('Time') +
    ylab(var) +
    theme_classic()
}

# Exclude timestamp from the loop
vars = gf_part_df %>%
  dplyr::select(-'TIMESTAMP_END') %>% 
  colnames()

for (i in vars) {
  p = plot_flux(gf_part_df, '2021-01-01', '2023-12-31', i)
  ggsave(filename = paste0('./60_MDS_REDDYPROC/', i, '_', run_id, '.png'), 
         plot = p, 
         dpi = 100,
         width = 12,
         height = 8)
} 


# Save file
write_csv(gf_part_df, './60_MDS_REDDYPROC/61.1_NEE_LE_REddyProcGapfilledPartitioned.csv')
