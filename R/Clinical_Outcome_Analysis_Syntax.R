##################### #
### Analysis of 
### Clinical measures
### for cTBS in OCD
### trial ID:
### ACTRN12616001687482

## Biostatisitician: Lachlan Webb

## libraries
library(readxl)
library(dplyr)
library(magrittr)
library(ggplot2)
library(tidyr)
library(rstatix)
library(emmeans)

# read in datasets
setwd("/Location/Of/Files")
# prepost data set has only measures for baseline 
# and post 4 weeks of cTBS - 50 observations
prepost_df <- read.csv("prepost_df.csv")
# follow up includes 6 month follow up measures
# due to drop out - 46 observations
follow_up_df <- read.csv("follow_up_df.csv")

YBOCS_df <- read.csv("YBOCS_only_df.csv")

options(pillar.sigfig = 7)


##### ##### ##### ##### ##### ##### ##### ##### #
##### ##### ##### ##### ##### ##### ##### ##### #
#### demographic and patient characteristics ####
####      Table 1 - Baseline demographic and    #
####      clinical characteristics              #
##### ##### ##### ##### ##### ##### ##### ##### #
##### ##### ##### ##### ##### ##### ##### ##### #
prepost_df %>% select(Age:AUDITSRtotal) %>% summary()
# Age
prepost_df %$% hist(Age)
prepost_df %$% summary(Age)
prepost_df %$% sd(Age)
prepost_df %>% group_by(Group) %>% summarise(meanAge = mean(Age), sdAge = sd(Age))
# Gender
prepost_df %>% mutate(Gender = recode_factor(Gender.F.1.M.2., `1` = "Female", `2` = "Male")) %$% table(Gender)
prepost_df %>% mutate(Gender = recode_factor(Gender.F.1.M.2., `1` = "Female", `2` = "Male")) %$% table(Gender)/sum(!is.na(prepost_df$Gender.F.1.M.2.))
prepost_df %>% mutate(Gender = recode_factor(Gender.F.1.M.2., `1` = "Female", `2` = "Male")) %>% filter(Group == "Active") %$% table(Gender)
prepost_df %>% mutate(Gender = recode_factor(Gender.F.1.M.2., `1` = "Female", `2` = "Male")) %>% filter(Group == "Active") %$% table(Gender)/sum(prepost_df %>% mutate(Gender = recode_factor(Gender.F.1.M.2., `1` = "Female", `2` = "Male")) %>% filter(Group == "Active") %$% table(Gender))
prepost_df %>% mutate(Gender = recode_factor(Gender.F.1.M.2., `1` = "Female", `2` = "Male")) %>% filter(Group == "Placebo") %$% table(Gender)
prepost_df %>% mutate(Gender = recode_factor(Gender.F.1.M.2., `1` = "Female", `2` = "Male")) %>% filter(Group == "Placebo") %$% table(Gender)/sum(prepost_df %>% mutate(Gender = recode_factor(Gender.F.1.M.2., `1` = "Female", `2` = "Male")) %>% filter(Group == "Placebo") %$% table(Gender))
# Handedness
prepost_df %>% mutate(Hand = recode_factor(Handedness.R.1.L.2., `1` = "Right", `2` = "Left")) %$% table(Hand)
prepost_df %>% mutate(Hand = recode_factor(Handedness.R.1.L.2., `1` = "Right", `2` = "Left")) %$% table(Hand)/sum(!is.na(prepost_df$Handedness.R.1.L.2.))
prepost_df %>% mutate(Hand = recode_factor(Handedness.R.1.L.2., `1` = "Right", `2` = "Left")) %>% filter(Group == "Active") %$% table(Hand)
prepost_df %>% mutate(Hand = recode_factor(Handedness.R.1.L.2., `1` = "Right", `2` = "Left")) %>% filter(Group == "Active") %$% table(Hand)/sum(prepost_df %>% mutate(Hand = recode_factor(Handedness.R.1.L.2., `1` = "Right", `2` = "Left")) %>% filter(Group == "Active") %$% table(Hand))
prepost_df %>% mutate(Hand = recode_factor(Handedness.R.1.L.2., `1` = "Right", `2` = "Left")) %>% filter(Group == "Placebo") %$% table(Hand)
prepost_df %>% mutate(Hand = recode_factor(Handedness.R.1.L.2., `1` = "Right", `2` = "Left")) %>% filter(Group == "Placebo") %$% table(Hand)/sum(prepost_df %>% mutate(Hand = recode_factor(Handedness.R.1.L.2., `1` = "Right", `2` = "Left")) %>% filter(Group == "Placebo") %$% table(Hand))
# Years of Education 
prepost_df %$% hist(Years_of_Education)
prepost_df %$% summary(Years_of_Education)
prepost_df %>% summarise(medianEdu = median(Years_of_Education), IQ1Edu = quantile(Years_of_Education,0.25), IQ3Edu = quantile(Years_of_Education,0.75))
prepost_df %>% group_by(Group) %>% summarise(medianEdu = median(Years_of_Education), IQ1Edu = quantile(Years_of_Education,0.25), IQ3Edu = quantile(Years_of_Education,0.75))
# Age of first Experience
prepost_df %>% filter(Age_of_First_Experience_script != 9999) %$% table(Group)
prepost_df %>% filter(Age_of_First_Experience_script != 9999) %$% hist(Age_of_First_Experience_script)
prepost_df %>% filter(Age_of_First_Experience_script != 9999) %$% summary(Age_of_First_Experience_script)
prepost_df %>% filter(Age_of_First_Experience_script != 9999) %>% summarise(medianExp = median(Age_of_First_Experience_script), IQ1Exp = quantile(Age_of_First_Experience_script,0.25), IQ3Exp = quantile(Age_of_First_Experience_script,0.75))
prepost_df %>% filter(Age_of_First_Experience_script != 9999) %>% group_by(Group) %>% summarise(medianExp = median(Age_of_First_Experience_script), IQ1Exp = quantile(Age_of_First_Experience_script,0.25), IQ3Exp = quantile(Age_of_First_Experience_script,0.75))
# Age of Diagnoses
prepost_df %>% filter(Age_of_Diagnosis_script != 9999) %$% table(Group)
prepost_df %>% filter(Age_of_Diagnosis_script != 9999) %$% hist(Age_of_Diagnosis_script)
prepost_df %>% filter(Age_of_Diagnosis_script != 9999) %$% summary(Age_of_Diagnosis_script)
prepost_df %>% filter(Age_of_Diagnosis_script != 9999) %>% summarise(medianDiag = median(Age_of_Diagnosis_script), IQ1Diag = quantile(Age_of_Diagnosis_script,0.25), IQ3Diag = quantile(Age_of_Diagnosis_script,0.75))
prepost_df %>% filter(Age_of_Diagnosis_script != 9999) %>% group_by(Group) %>% summarise(medianDiag = median(Age_of_Diagnosis_script), IQ1Diag = quantile(Age_of_Diagnosis_script,0.25), IQ3Diag = quantile(Age_of_Diagnosis_script,0.75))

# FSIQ2
prepost_df %$% table(Group)
prepost_df %$% hist(FSIQ.2_Comp_Score)
prepost_df %$% summary(FSIQ.2_Comp_Score)
prepost_df %$% sd(FSIQ.2_Comp_Score)
prepost_df %>% group_by(Group) %>% summarise(meanEIQ2 = mean(FSIQ.2_Comp_Score), sdIQ2 = sd(FSIQ.2_Comp_Score))

# FSIQ 4
prepost_df %$% table(Group)
prepost_df %$% hist(FSIQ.4_Comp_Score)
prepost_df %$% summary(FSIQ.4_Comp_Score)
prepost_df %$% sd(FSIQ.4_Comp_Score)
prepost_df %>% group_by(Group) %>% summarise(meanEIQ4 = mean(FSIQ.4_Comp_Score), sdIQ4 = sd(FSIQ.4_Comp_Score))


######

##### ##### ##### ##### ##### #### #
##### ##### ##### ##### ##### #### #
#### outcome summaries measures ####
####  eTable 2 - Outcome summaries #
##### ##### ##### ##### ##### #### #
##### ##### ##### ##### ##### #### #

# mean and SD for all measures at 
# each time point and difference 
# between time points
# normality already assessed to 
# not be violated

options(pillar.sigfig = 7) 

## including incomplete cases N=50, prepost_df has n=50 ####
# this is eTable 2

prepost_df %>% merge(follow_up_df %>% select(Participant_ID, YBOCS_Total_6mnth, YBOCS_delta_13), all = TRUE) %>% 
  select(YBOCS_Total_Pre,YBOCS_Total_Post,YBOCS_Total_6mnth, YBOCS_delta_12, YBOCS_delta_12, YBOCS_delta_13) %>% 
  pivot_longer(cols = everything(), names_to = "Measure", values_to = "value") %>%
  drop_na() %>%
  group_by(Measure) %>% 
  summarise(mean = mean(value), sd = sd(value), n= n())
prepost_df %>% merge(follow_up_df %>% select(Participant_ID, Group, YBOCS_Total_6mnth, YBOCS_delta_13), all = TRUE) %>% 
  select(Group, YBOCS_Total_Pre,YBOCS_Total_Post,YBOCS_Total_6mnth, YBOCS_delta_12, YBOCS_delta_12, YBOCS_delta_13) %>% 
  pivot_longer(cols = !Group, names_to = "Measure", values_to = "value") %>%
  drop_na() %>%
  group_by(Measure, Group) %>% 
  summarise(mean = mean(value), sd = sd(value), n= n())
prepost_df %>% select(Group, YBOCS_Total_Pre,YBOCS_Total_Post) %>%
  filter(Group == "Active") %$%
  lsr::cohensD(YBOCS_Total_Pre,YBOCS_Total_Post)
follow_up_df %>% filter(!is.na(YBOCS_Total_6mnth)) %>% select(Group, YBOCS_Total_Pre,YBOCS_Total_6mnth) %>%
  filter(Group == "Active") %$%
  lsr::cohensD(YBOCS_Total_Pre,YBOCS_Total_6mnth)
prepost_df %>% select(Group, YBOCS_Total_Pre,YBOCS_Total_Post) %>%
  filter(Group == "Placebo") %$%
  lsr::cohensD(YBOCS_Total_Pre,YBOCS_Total_Post)
follow_up_df %>% filter(!is.na(YBOCS_Total_6mnth)) %>% select(Group, YBOCS_Total_Pre,YBOCS_Total_6mnth) %>%
  filter(Group == "Placebo") %$%
  lsr::cohensD(YBOCS_Total_Pre,YBOCS_Total_6mnth)
prepost_df %>% t.test(YBOCS_Total_Pre ~ Group, data = ., var.equal = TRUE)
prepost_df %>% t.test(YBOCS_delta_12 ~ Group, data = ., var.equal = TRUE)
follow_up_df %>% t.test(YBOCS_delta_13 ~ Group, data = ., var.equal = TRUE)

# OCIR
prepost_df %>% merge(follow_up_df %>% select(Participant_ID, OCIR_Total_6mnth, OCIR_delta_13), all = TRUE) %>% 
  select(OCIR_Total_Pre,OCIR_Total_Post,OCIR_Total_6mnth, OCIR_delta_12, OCIR_delta_13) %>% 
  pivot_longer(cols = everything(), names_to = "Measure", values_to = "value") %>%
  drop_na() %>%
  group_by(Measure) %>% 
  summarise(mean = mean(value), sd = sd(value), n= n())
prepost_df %>% merge(follow_up_df %>% select(Participant_ID, Group, OCIR_Total_6mnth, OCIR_delta_13), all = TRUE) %>% 
  select(Group, OCIR_Total_Pre,OCIR_Total_Post,OCIR_Total_6mnth, OCIR_delta_12, OCIR_delta_13) %>% 
  pivot_longer(cols = !Group, names_to = "Measure", values_to = "value") %>%
  drop_na() %>%
  group_by(Measure, Group) %>% 
  summarise(mean = mean(value), sd = sd(value), n= n())
prepost_df %>% t.test(OCIR_Total_Pre ~ Group, data = ., var.equal = TRUE)
prepost_df %>% t.test(OCIR_delta_12 ~ Group, data = ., var.equal = TRUE)
follow_up_df %>% t.test(OCIR_delta_13 ~ Group, data = ., var.equal = TRUE)

# OBQ
prepost_df %>% merge(follow_up_df %>% select(Participant_ID, OBQ_Total_6mnth, OBQ_Total_delta_13), all = TRUE) %>% 
  select(OBQ_Total_Pre,OBQ_Total_Post,OBQ_Total_6mnth, OBQ_Total_delta_12, OBQ_Total_delta_13) %>% 
  pivot_longer(cols = everything(), names_to = "Measure", values_to = "value") %>%
  drop_na() %>%
  group_by(Measure) %>% 
  summarise(mean = mean(value), sd = sd(value), n= n())
prepost_df %>% merge(follow_up_df %>% select(Participant_ID, Group, OBQ_Total_6mnth, OBQ_Total_delta_13), all = TRUE) %>% 
  select(Group, OBQ_Total_Pre,OBQ_Total_Post,OBQ_Total_6mnth, OBQ_Total_delta_12, OBQ_Total_delta_13) %>% 
  pivot_longer(cols = !Group, names_to = "Measure", values_to = "value") %>%
  drop_na() %>%
  group_by(Measure, Group) %>% 
  summarise(mean = mean(value), sd = sd(value), n= n())
prepost_df %>% t.test(OBQ_Total_Pre ~ Group, data = ., var.equal = TRUE)
prepost_df %>% t.test(OBQ_Total_delta_12 ~ Group, data = ., var.equal = TRUE)
follow_up_df %>% t.test(OBQ_Total_delta_13 ~ Group, data = ., var.equal = TRUE)

# HAMA
prepost_df %>% merge(follow_up_df %>% select(Participant_ID, HAMA_Total_6mnth, HAMA_delta_13), all = TRUE) %>% 
  select(HAMA_Total_Pre,HAMA_Total_Post,HAMA_Total_6mnth, HAMA_delta_12, HAMA_delta_13) %>% 
  pivot_longer(cols = everything(), names_to = "Measure", values_to = "value") %>%
  drop_na() %>%
  group_by(Measure) %>% 
  summarise(mean = mean(value), sd = sd(value), n= n())
prepost_df %>% merge(follow_up_df %>% select(Participant_ID, Group, HAMA_Total_6mnth, HAMA_delta_13), all = TRUE) %>% 
  select(Group, HAMA_Total_Pre,HAMA_Total_Post,HAMA_Total_6mnth, HAMA_delta_12, HAMA_delta_13) %>% 
  pivot_longer(cols = !Group, names_to = "Measure", values_to = "value") %>%
  drop_na() %>%
  group_by(Measure, Group) %>% 
  summarise(mean = mean(value), sd = sd(value), n= n())
prepost_df %>% t.test(HAMA_Total_Pre ~ Group, data = ., var.equal = TRUE)
prepost_df %>% t.test(HAMA_delta_12 ~ Group, data = ., var.equal = TRUE)
follow_up_df %>% t.test(HAMA_delta_13 ~ Group, data = ., var.equal = TRUE)

# HADS Anxiety
prepost_df %>% merge(follow_up_df %>% select(Participant_ID, Anx_total_6mnth, HADS_ANX_delta_13), all = TRUE) %>% 
  select(Anx_total_Pre,Anx_total_Post,Anx_total_6mnth, HADS_ANX_delta_12, HADS_ANX_delta_13) %>% 
  pivot_longer(cols = everything(), names_to = "Measure", values_to = "value") %>%
  drop_na() %>%
  group_by(Measure) %>% 
  summarise(mean = mean(value), sd = sd(value), n= n())
prepost_df %>% merge(follow_up_df %>% select(Participant_ID, Group, Anx_total_6mnth, HADS_ANX_delta_13), all = TRUE) %>% 
  select(Group, Anx_total_Pre,Anx_total_Post,Anx_total_6mnth, HADS_ANX_delta_12, HADS_ANX_delta_13) %>% 
  pivot_longer(cols = !Group, names_to = "Measure", values_to = "value") %>%
  drop_na() %>%
  group_by(Measure, Group) %>% 
  summarise(mean = mean(value), sd = sd(value), n= n())
prepost_df %>% t.test(Anx_total_Pre ~ Group, data = ., var.equal = TRUE)
prepost_df %>% t.test(HADS_ANX_delta_12 ~ Group, data = ., var.equal = TRUE)
follow_up_df %>% t.test(HADS_ANX_delta_13 ~ Group, data = ., var.equal = TRUE)

# MADRS
prepost_df %>% merge(follow_up_df %>% select(Participant_ID, MADRS_Total_6mnth, MADRS_delta_13), all = TRUE) %>% 
  select(MADRS_Total_Pre,MADRS_Total_Post,MADRS_Total_6mnth, MADRS_delta_12, MADRS_delta_13) %>% 
  pivot_longer(cols = everything(), names_to = "Measure", values_to = "value") %>%
  drop_na() %>%
  group_by(Measure) %>% 
  summarise(mean = mean(value), sd = sd(value), n= n())
prepost_df %>% merge(follow_up_df %>% select(Participant_ID, Group, MADRS_Total_6mnth, MADRS_delta_13), all = TRUE) %>% 
  select(Group, MADRS_Total_Pre,MADRS_Total_Post,MADRS_Total_6mnth, MADRS_delta_12, MADRS_delta_13) %>% 
  pivot_longer(cols = !Group, names_to = "Measure", values_to = "value") %>%
  drop_na() %>%
  group_by(Measure, Group) %>% 
  summarise(mean = mean(value), sd = sd(value), n= n())
prepost_df %>% t.test(MADRS_Total_Pre ~ Group, data = ., var.equal = TRUE)
prepost_df %>% t.test(MADRS_delta_12 ~ Group, data = ., var.equal = TRUE)
follow_up_df %>% t.test(MADRS_delta_13 ~ Group, data = ., var.equal = TRUE)

# HADS Depression
prepost_df %>% merge(follow_up_df %>% select(Participant_ID, Dep_Total_6mnth, HADS_DEP_delta_13), all = TRUE) %>% 
  select(Dep_Total_Pre,Dep_Total_Post,Dep_Total_6mnth, HADS_DEP_delta_12, HADS_DEP_delta_13) %>% 
  pivot_longer(cols = everything(), names_to = "Measure", values_to = "value") %>%
  drop_na() %>%
  group_by(Measure) %>% 
  summarise(mean = mean(value), sd = sd(value), n= n())
prepost_df %>% merge(follow_up_df %>% select(Participant_ID, Group, Dep_Total_6mnth, HADS_DEP_delta_13), all = TRUE) %>% 
  select(Group, Dep_Total_Pre,Dep_Total_Post,Dep_Total_6mnth, HADS_DEP_delta_12, HADS_DEP_delta_13) %>% 
  pivot_longer(cols = !Group, names_to = "Measure", values_to = "value") %>%
  drop_na() %>%
  group_by(Measure, Group) %>% 
  summarise(mean = mean(value), sd = sd(value), n= n())
prepost_df %>% t.test(Dep_Total_Pre ~ Group, data = ., var.equal = TRUE)
prepost_df %>% t.test(HADS_DEP_delta_12 ~ Group, data = ., var.equal = TRUE)
follow_up_df %>% t.test(HADS_DEP_delta_13 ~ Group, data = ., var.equal = TRUE)




##### ##### ##### ##### ##### ##### ##### ## #
##### ##### ##### ##### ##### ##### ##### ## #
#### 2 time point repeated measures ANOVA ####
####    eTable 3 - Mixed ANOVA results    ## #
####                                      ## #
####    eTable 4 - interaction contrast   ## #
####    and within group contrast         ## #
##### ##### ##### ##### ##### ##### ##### ## #
##### ##### ##### ##### ##### ##### ##### ## #

## Primary Outcome - YBOCS ####

YBOCSaov_df <- prepost_df %>% 
  select(Participant_ID, Group, YBOCS_Total_Pre, YBOCS_Total_Post) %>%
  pivot_longer(cols = YBOCS_Total_Pre:YBOCS_Total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID))

# overall repeated measures/mixed anova 
YBOCSaov <- aov(score ~ Group*Time + Error(Participant_ID), data = YBOCSaov_df)
# summary of ANOVA 
YBOCSaov %>% summary()
# overall repeated measures/mixed anova - get more descriptives (eta squared)
YBOCSaov %>%  DescTools::EtaSq(type = 1, anova = TRUE)

# interaction contrast
condmeans_YBOCS <- emmeans(YBOCSaov, ~ Group * Time)
contrast(condmeans_YBOCS, list("interaction"= c(1,-1,-1,1)))
confint(contrast(condmeans_YBOCS, list("interaction"= c(1,-1,-1,1))))

# within group contrast
prepost_df %>% 
  select(Participant_ID, Group, YBOCS_Total_Pre, YBOCS_Total_Post) %>%
  pivot_longer(cols = YBOCS_Total_Pre:YBOCS_Total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID))  %>% 
  group_by(Group) %>% pairwise_t_test(score ~ Time, paired  = TRUE, detailed = TRUE, ref.group = "Post")

### YBOCS subscales

# pre post repeated measures anova
RMAOV <- YBOCS_df %>% select(Participant_ID, Time, Group, OBSST) %>%
  mutate(score = as.numeric(OBSST)) %>%
  filter(Time %in% c("Pre","Post")) %>%
  drop_na() %>%
  group_by(Participant_ID) %>% mutate(num = n()) %>% filter(num == 2) %>% ungroup() %>%
  mutate(Time = factor(Time, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group*Time + Error(Participant_ID), data = .)
RMAOV %>% summary()
RMAOV %>% DescTools::EtaSq(type = 1, anova = TRUE)

# pre post repeated measures anova
RMAOV <- YBOCS_df %>% select(Participant_ID, Time, Group, COMST) %>%
  mutate(score = as.numeric(COMST)) %>%
  filter(Time %in% c("Pre","Post")) %>%
  drop_na() %>%
  group_by(Participant_ID) %>% mutate(num = n()) %>% filter(num == 2) %>% ungroup() %>%
  mutate(Time = factor(Time, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group*Time + Error(Participant_ID), data = .)
RMAOV %>% summary()
RMAOV %>% DescTools::EtaSq(type = 1, anova = TRUE)



## Secondary Outcomes ####

## OCIR ####
prepost_df %>% 
  select(Participant_ID, Group, OCIR_Total_Pre, OCIR_Total_Post) %>%
  pivot_longer(cols = OCIR_Total_Pre:OCIR_Total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% summary()


prepost_df %>% 
  select(Participant_ID, Group, OCIR_Total_Pre, OCIR_Total_Post) %>%
  pivot_longer(cols = OCIR_Total_Pre:OCIR_Total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% DescTools::EtaSq(type = 1, anova = TRUE)
## OBQ ####
prepost_df %>% 
  select(Participant_ID, Group, OBQ_Total_Pre, OBQ_Total_Post) %>%
  pivot_longer(cols = OBQ_Total_Pre:OBQ_Total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% summary()
prepost_df %>% 
  select(Participant_ID, Group, OBQ_Total_Pre, OBQ_Total_Post) %>%
  pivot_longer(cols = OBQ_Total_Pre:OBQ_Total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .)%>% DescTools::EtaSq(type = 1, anova = TRUE)

## HAM-A ####
## include contrasts

# overall mixed anova
HAMAaov_df <- prepost_df %>% 
  select(Participant_ID, Group, HAMA_Total_Pre, HAMA_Total_Post) %>%
  pivot_longer(cols = HAMA_Total_Pre:HAMA_Total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) 

prepost_df %>% 
  select(Participant_ID, Group, HAMA_Total_Pre, HAMA_Total_Post) %>%
  pivot_longer(cols = HAMA_Total_Pre:HAMA_Total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% summary()
prepost_df %>% 
  select(Participant_ID, Group, HAMA_Total_Pre, HAMA_Total_Post) %>%
  pivot_longer(cols = HAMA_Total_Pre:HAMA_Total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% DescTools::EtaSq(type = 1, anova = TRUE)

# aov object for contrasts
HAMA_aov <- aov(score ~ Group * Time + Error(Participant_ID), data = HAMAaov_df)

# interaction contrast
condmeans_HAMA <- emmeans(HAMA_aov, ~ Group * Time)
contrast(condmeans_HAMA, list("interaction"= c(1,-1,-1,1)))
confint(contrast(condmeans_HAMA, list("interaction"= c(1,-1,-1,1))))

# within group contrasts
prepost_df %>% 
  select(Participant_ID, Group, HAMA_Total_Pre, HAMA_Total_Post) %>%
  pivot_longer(cols = HAMA_Total_Pre:HAMA_Total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  group_by(Group) %>% pairwise_t_test(score ~ Time, paired  = TRUE, detailed = TRUE, ref.group = "Post")

## HADS Anxiety ####
prepost_df %>% 
  select(Participant_ID, Group, Anx_total_Pre, Anx_total_Post) %>%
  pivot_longer(cols = Anx_total_Pre:Anx_total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% summary()

prepost_df %>% 
  select(Participant_ID, Group, Anx_total_Pre, Anx_total_Post) %>%
  pivot_longer(cols = Anx_total_Pre:Anx_total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% DescTools::EtaSq(type = 1, anova = TRUE)

## MADRS ####
## include contrasts

# overal mixed anova
prepost_df %>% 
  select(Participant_ID, Group, MADRS_Total_Pre, MADRS_Total_Post) %>%
  pivot_longer(cols = MADRS_Total_Pre:MADRS_Total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% summary()
prepost_df %>% 
  select(Participant_ID, Group, MADRS_Total_Pre, MADRS_Total_Post) %>%
  pivot_longer(cols = MADRS_Total_Pre:MADRS_Total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% DescTools::EtaSq(type = 1, anova = TRUE)


# aov object for contrasts
MADRSaov_df <- prepost_df %>% 
  select(Participant_ID, Group, MADRS_Total_Pre, MADRS_Total_Post) %>%
  pivot_longer(cols = MADRS_Total_Pre:MADRS_Total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID))

MADRS_aov <- aov(score ~ Group * Time + Error(Participant_ID), data = MADRSaov_df)

# interaction contrast
condmeans_MADRS <- emmeans(MADRS_aov, ~ Group * Time)
contrast(condmeans_MADRS, list("interaction"= c(1,-1,-1,1)))
confint(contrast(condmeans_MADRS, list("interaction"= c(1,-1,-1,1))))

# within group contrasts
prepost_df %>% 
  select(Participant_ID, Group, MADRS_Total_Pre, MADRS_Total_Post) %>%
  pivot_longer(cols = MADRS_Total_Pre:MADRS_Total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  group_by(Group) %>% pairwise_t_test(score ~ Time, paired  = TRUE, detailed = TRUE, ref.group = "Post")


## HADS Depression ####
prepost_df %>% 
  select(Participant_ID, Group, Dep_Total_Pre, Dep_Total_Post) %>%
  pivot_longer(cols = Dep_Total_Pre:Dep_Total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% summary()

prepost_df %>% 
  select(Participant_ID, Group, Dep_Total_Pre, Dep_Total_Post) %>%
  pivot_longer(cols = Dep_Total_Pre:Dep_Total_Post, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post"), labels = c("Pre","Post"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .)%>% DescTools::EtaSq(type = 1, anova = TRUE)







##### ##### ##### ##### ##### ##### ##### ## #
##### ##### ##### ##### ##### ##### ##### ## #
#### 3 time point repeated measures ANOVA ####
####    eTable 3 - Mixed ANOVA results    ## #
##### ##### ##### ##### ##### ##### ##### ## #
##### ##### ##### ##### ##### ##### ##### ## #

# Repeated Measures ANOVA with interaction

## Primary Outcome - YBOCS ####
follow_up_df %>% filter(!is.na(YBOCS_Total_6mnth)) %>%
  select(Participant_ID, Group, YBOCS_Total_Pre, YBOCS_Total_Post, YBOCS_Total_6mnth) %>%
  pivot_longer(cols = YBOCS_Total_Pre:YBOCS_Total_6mnth, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post","6mnth"), labels = c("Pre","Post","Final"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group*Time + Error(Participant_ID), data = .) %>% summary()

follow_up_df %>% filter(!is.na(YBOCS_Total_6mnth)) %>%
  select(Participant_ID, Group, YBOCS_Total_Pre, YBOCS_Total_Post, YBOCS_Total_6mnth) %>%
  pivot_longer(cols = YBOCS_Total_Pre:YBOCS_Total_6mnth, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post","6mnth"), labels = c("Pre","Post","Final"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group*Time + Error(Participant_ID), data = .) %>% DescTools::EtaSq(type = 1, anova = TRUE)

# OBSST ####

RMAOV3 <- YBOCS_df %>% select(Participant_ID, Time, Group, OBSST) %>%
  mutate(score = as.numeric(OBSST)) %>%
  filter(Time %in% c("Pre","Post","6mnth")) %>%
  drop_na() %>%
  group_by(Participant_ID) %>% mutate(num = n()) %>% filter(num == 3) %>% ungroup() %>%
  mutate(Time = factor(Time, levels = c("Pre","Post","6mnth"), labels = c("Pre","Post","Final"))) %>%
  mutate(Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group*Time + Error(Participant_ID), data = .)
RMAOV3 %>% summary()
RMAOV3 %>% DescTools::EtaSq(type = 1, anova = TRUE)

# COMST  ####

# 3 time point repeated measures anova
RMAOV3 <- YBOCS_df %>% select(Participant_ID, Time, Group, COMST) %>%
  mutate(score = as.numeric(COMST)) %>%
  filter(Time %in% c("Pre","Post","6mnth")) %>%
  drop_na() %>%
  group_by(Participant_ID) %>% mutate(num = n()) %>% filter(num == 3) %>% ungroup() %>%
  mutate(Time = factor(Time, levels = c("Pre","Post","6mnth"), labels = c("Pre","Post","Final"))) %>%
  mutate(Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group*Time + Error(Participant_ID), data = .)
RMAOV3 %>% summary()
RMAOV3 %>% DescTools::EtaSq(type = 1, anova = TRUE)


## Secondary Outcomes ####

## OCIR ####

follow_up_df %>% filter(!is.na(OCIR_Total_6mnth)) %>%
  select(Participant_ID, Group, OCIR_Total_Pre, OCIR_Total_Post, OCIR_Total_6mnth) %>%
  pivot_longer(cols = OCIR_Total_Pre:OCIR_Total_6mnth, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post","6mnth"), labels = c("Pre","Post","Final"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% summary()

follow_up_df %>% filter(!is.na(OCIR_Total_6mnth)) %>%
  select(Participant_ID, Group, OCIR_Total_Pre, OCIR_Total_Post, OCIR_Total_6mnth) %>%
  pivot_longer(cols = OCIR_Total_Pre:OCIR_Total_6mnth, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post","6mnth"), labels = c("Pre","Post","Final"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% DescTools::EtaSq(type = 1, anova = TRUE)

## OBQ ####

follow_up_df %>% filter(!is.na(OBQ_Total_6mnth)) %>%
  select(Participant_ID, Group, OBQ_Total_Pre, OBQ_Total_Post, OBQ_Total_6mnth) %>%
  pivot_longer(cols = OBQ_Total_Pre:OBQ_Total_6mnth, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post","6mnth"), labels = c("Pre","Post","Final"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% summary()

follow_up_df %>% filter(!is.na(OBQ_Total_6mnth)) %>%
  select(Participant_ID, Group, OBQ_Total_Pre, OBQ_Total_Post, OBQ_Total_6mnth) %>%
  pivot_longer(cols = OBQ_Total_Pre:OBQ_Total_6mnth, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post","6mnth"), labels = c("Pre","Post","Final"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% DescTools::EtaSq(type = 1, anova = TRUE)

## HAM-A ####

follow_up_df %>% filter(!is.na(HAMA_Total_6mnth)) %>%
  select(Participant_ID, Group, HAMA_Total_Pre, HAMA_Total_Post, HAMA_Total_6mnth) %>%
  pivot_longer(cols = HAMA_Total_Pre:HAMA_Total_6mnth, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post","6mnth"), labels = c("Pre","Post","Final"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% summary()
follow_up_df %>% filter(!is.na(HAMA_Total_6mnth)) %>%
  select(Participant_ID, Group, HAMA_Total_Pre, HAMA_Total_Post, HAMA_Total_6mnth) %>%
  pivot_longer(cols = HAMA_Total_Pre:HAMA_Total_6mnth, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post","6mnth"), labels = c("Pre","Post","Final"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% DescTools::EtaSq(type = 1, anova = TRUE)

## HADS Anxiety ####

follow_up_df %>% filter(!is.na(Anx_total_6mnth)) %>%
  select(Participant_ID, Group, Anx_total_Pre, Anx_total_Post, Anx_total_6mnth) %>%
  pivot_longer(cols = Anx_total_Pre:Anx_total_6mnth, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post","6mnth"), labels = c("Pre","Post","Final"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% summary()

follow_up_df %>% filter(!is.na(Anx_total_6mnth)) %>%
  select(Participant_ID, Group, Anx_total_Pre, Anx_total_Post, Anx_total_6mnth) %>%
  pivot_longer(cols = Anx_total_Pre:Anx_total_6mnth, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post","6mnth"), labels = c("Pre","Post","Final"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% DescTools::EtaSq(type = 1, anova = TRUE)

## MADRS ####

follow_up_df %>% filter(!is.na(MADRS_Total_6mnth)) %>%
  select(Participant_ID, Group, MADRS_Total_Pre, MADRS_Total_Post, MADRS_Total_6mnth) %>%
  pivot_longer(cols = MADRS_Total_Pre:MADRS_Total_6mnth, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post","6mnth"), labels = c("Pre","Post","Final"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% summary()

follow_up_df %>% filter(!is.na(MADRS_Total_6mnth)) %>%
  select(Participant_ID, Group, MADRS_Total_Pre, MADRS_Total_Post, MADRS_Total_6mnth) %>%
  pivot_longer(cols = MADRS_Total_Pre:MADRS_Total_6mnth, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post","6mnth"), labels = c("Pre","Post","Final"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% DescTools::EtaSq(type = 1, anova = TRUE)



## HADS Depression ####

follow_up_df %>% filter(!is.na(Dep_Total_6mnth)) %>%
  select(Participant_ID, Group, Dep_Total_Pre, Dep_Total_Post, Dep_Total_6mnth) %>%
  pivot_longer(cols = Dep_Total_Pre:Dep_Total_6mnth, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post","6mnth"), labels = c("Pre","Post","Final"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% summary()
follow_up_df %>% filter(!is.na(Dep_Total_6mnth)) %>%
  select(Participant_ID, Group, Dep_Total_Pre, Dep_Total_Post, Dep_Total_6mnth) %>%
  pivot_longer(cols = Dep_Total_Pre:Dep_Total_6mnth, names_to = "Measure", values_to = "score") %>%
  separate(Measure,into = c("Questionnaire","Total","TimeStage"), "_", remove = FALSE) %>%
  mutate(Time = factor(TimeStage, levels = c("Pre","Post","6mnth"), labels = c("Pre","Post","Final"))) %>%
  mutate(score = as.numeric(score), 
         Group  = factor(Group, levels = c("Placebo","Active"), labels = c("Placebo","Active")), 
         Participant_ID = as.factor(Participant_ID)) %>% 
  aov(score ~ Group * Time + Error(Participant_ID), data = .) %>% DescTools::EtaSq(type = 1, anova = TRUE)






