##################### #
### Figures of 
### Clinical measures
### for cTBS in OCD
### trial ID:
### ACTRN12616001687482

## Biostatisitician: Lachlan Webb

setwd("/Location/Of/Files")
library(readxl)
library(dplyr)
library(magrittr)
library(ggplot2)
library(tidyr)
library(RColorBrewer)
library(car)
library(ggpubr)
library(rlang)
library(ggExtra)
library(ggsci)
# follow up includes 6 month follow up measures
# due to drop out - 46 observations
follow_up_df <- read.csv("follow_up_df.csv")
# prepost data set has only measures for baseline 
# and post 4 weeks of cTBS - 50 observations
prepost_df <- read.csv("prepost_df.csv")


#### #### #### #### #### #
#### eFigure 3      ####
#### #### #### #### #### #


prepost_df %>% merge(follow_up_df, all = TRUE) %>%
  select(Group, YBOCS_delta_12, OCIR_delta_12, OBQ_Total_delta_12, HAMA_delta_12, MADRS_delta_12, HADS_ANX_delta_12, HADS_DEP_delta_12, 
         YBOCS_delta_13, OCIR_delta_13, OBQ_Total_delta_13, HAMA_delta_13, MADRS_delta_13, HADS_ANX_delta_13, HADS_DEP_delta_13) %>%
  pivot_longer(cols = YBOCS_delta_12:HADS_DEP_delta_13, names_to = "Questionnaire_change", values_to = "value") %>%
  group_by(Questionnaire_change, Group) %>%
  mutate(mean_diff = mean(value, na.rm = TRUE),
         samsiz = paste("n =",sum(!is.na(value)))) %>%
  ungroup() %>% 
  mutate(Questionnaire_change = dplyr::recode(Questionnaire_change,HADS_ANX_delta_12 = "HADS-ANX_delta_12",HADS_ANX_delta_13 = "HADS-ANX_delta_13",
                                              HADS_DEP_delta_12 = "HADS-DEP_delta_12",HADS_DEP_delta_13 = "HADS-DEP_delta_13",
                                              OBQ_Total_delta_12 = "OBQ_delta_12",OBQ_Total_delta_13  = "OBQ_delta_13")) %>%
  separate(col = Questionnaire_change, into = c("Questionnaire","delta","diff"), remove = FALSE, sep = "_") %>%
  mutate(Questionnaire_type_fac = factor(Questionnaire, 
                                         levels = c("YBOCS", "OCIR", "OBQ", "HAMA", "HADS-ANX", "MADRS", "HADS-DEP"),
                                         labels = c("Y-BOCS", "OCIR", "OBQ", "HAM-A", "HADS-Anx", "MADRS", "HADS-Dep"))) %>%
  mutate(period_diff = factor(diff, levels = c("12", "13"), labels = c("Post-cTBS - Baseline", "Follow-up - Baseline"))) %>%
  mutate(Group_fac = factor(Group,
                            levels = c("Placebo","Active"),
                            labels = c("Placebo","Active"))) %>%
  group_by(Questionnaire_type_fac) %>%
  mutate(maxlab = max(value, na.rm = TRUE) + (range(value, na.rm = TRUE)[2] - range(value, na.rm = TRUE)[1])*0.1) %>%
  ungroup() %>%
  ggplot(aes(x = Group_fac, y = value, colour = Group_fac)) + 
  geom_violin(fill = NA, alpha = 0.1, weight = 0.5, show.legend = FALSE) + 
  geom_jitter(height = 0, width = 0.1, alpha = 0.4, show.legend = FALSE) + 
  geom_segment(aes(x = as.numeric(Group_fac)-0.2, y = mean_diff, xend = as.numeric(Group_fac)+0.2, yend = mean_diff), size = 1.1, show.legend = FALSE) + 
  geom_text(aes(x = Group, y = maxlab, label =samsiz ), show.legend = FALSE, vjust = 1, hjust = -0.45, size = 2.2)  + 
  facet_grid(Questionnaire_type_fac~period_diff, scales = "free") + 
  ggsci::scale_color_jama() +
  theme_classic() + 
  labs(x = "", y = "Change in Questionnaire Score") + 
  scale_x_discrete(breaks = c("Placebo","Active"), labels = c("Sham","Active"))


#### #### #### #### #### #
#### eFigure 4      ####
#### #### #### #### #### #


prepost_df %>% select(Participant_ID,Group,YBOCS_Total_Pre,YBOCS_Total_Post,HAMA_Total_Pre,HAMA_Total_Post,
                      MADRS_Total_Pre,MADRS_Total_Post,OCIR_Total_Pre,OCIR_Total_Post,OBQ_Total_Pre,OBQ_Total_Post,
                      Anx_total_Pre,Anx_total_Post,Dep_Total_Pre,Dep_Total_Post) %>%
  merge(follow_up_df %>% select(Participant_ID,YBOCS_Total_6mnth,HAMA_Total_6mnth,MADRS_Total_6mnth,OCIR_Total_6mnth,
                                OBQ_Total_6mnth,Anx_total_6mnth,Dep_Total_6mnth), by = "Participant_ID") %>%
  pivot_longer(cols = YBOCS_Total_Pre:Dep_Total_6mnth, names_to = "Measure", values_to = "value") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post","6mnth"), labels = c("Pre","Post","Final"))) %>%
  mutate(Questionnaire_fac = factor(Questionnaire, 
                                    levels = c("YBOCS", "OCIR", "OBQ", "HAMA", "Anx", "MADRS", "Dep"),
                                    labels = c("Y-BOCS", "OCIR", "OBQ", "HAM-A", "HADS-Anx", "MADRS", "HADS-Dep"))) %>%
  mutate(Group_fac = factor(Group,
                            levels = c("Placebo","Active"),
                            labels = c("Sham","Active"))) %>%
  group_by(Participant_ID, Questionnaire_fac) %>%
  mutate(pre_value = ifelse(Time == "Pre",value,NA), final_value = ifelse(Time == "Final",value,NA)) %>%
  arrange(Participant_ID,Questionnaire_fac,Time) %>%
  fill(pre_value) %>% 
  fill(final_value, .direction  = "up") %>% 
  mutate(rel_value = value - pre_value, rel_final = final_value - pre_value) %>%
  ggplot(aes(x = Time, y = rel_value, colour = Group_fac, group = paste(Participant_ID,Questionnaire_fac), linetype = (rel_final<0))) + 
  geom_point() + 
  geom_line(size = 0.01) + 
  ggsci::scale_color_jama(name = "Group") +
  scale_linetype_manual(breaks = c(TRUE, FALSE), values = c("solid","dashed"), name = "Overall \nTrend", labels = c("Decrease","Increase or \nConstant")) +
  facet_grid(Questionnaire_fac~Group_fac, scales = "free") + 
  theme_bw() + 
  labs(x = "Time Point", y = "Change in Score", title = "Change in Questionnaire Response Over Time", subtitle = "Relative to Baseline") + 
  scale_x_discrete(breaks = c("Pre","Post","Final"), labels = c("Baseline","Post-cTBS","Follow-up"))




