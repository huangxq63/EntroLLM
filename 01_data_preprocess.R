# Data Preprocess
# Data Source:https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&


library(foreign)
library(dplyr)
library(tidyr)
library(openxlsx)
library(tidyverse)

# Import dataset.

# NHANES 2003-2004

bmx_d1 = read.xport("E:/EntroLLM/NHANES 2003-2004/BMX_C.xpt") %>% 
  drop_na(BMXBMI) %>% 
  select(SEQN,BMXBMI)

demo_d1 = read.xport("E:/EntroLLM/NHANES 2003-2004/DEMO_C.xpt") %>% 
  filter(RIDAGEYR>=20 & RIDAGEYR<85 & DMDMARTL!=77 & DMDMARTL!=99 & RIDEXPRG %in% c(2,3,NA) & DMDEDUC2!=7 & DMDEDUC2!=9) %>% 
  select(SEQN,RIAGENDR,RIDAGEYR,RIDRETH1,DMDEDUC2,DMDMARTL,INDFMPIR) %>% 
  drop_na(SEQN,RIAGENDR,RIDAGEYR,RIDRETH1,DMDEDUC2,DMDMARTL,INDFMPIR)

paxraw_d1 = read.xport("E:/EntroLLM/NHANES 2003-2004/paxraw_c.xpt") %>% 
  filter(PAXSTAT==1 & PAXCAL==1) %>% 
  drop_na(PAXINTEN) %>% 
  select(SEQN,PAXN,PAXINTEN) 

# NHANES 2005-2006

bmx_d2 = read.xport("E:/EntroLLM/NHANES 2005-2006/BMX_D.xpt") %>%
  drop_na(BMXBMI) %>% 
  select(SEQN,BMXBMI)

demo_d2 = read.xport("E:/EntroLLM/NHANES 2005-2006/DEMO_D.xpt") %>% 
  filter(RIDAGEYR>=20 & RIDAGEYR<85 & DMDMARTL!=77 & DMDMARTL!=99 & RIDEXPRG %in% c(2,3,NA) & DMDEDUC2!=7 & DMDEDUC2!=9) %>% 
  select(SEQN,RIAGENDR,RIDAGEYR,RIDRETH1,DMDEDUC2,DMDMARTL,INDFMPIR) %>% 
  drop_na(SEQN,RIAGENDR,RIDAGEYR,RIDRETH1,DMDEDUC2,DMDMARTL,INDFMPIR)

paxraw_d2 = read.xport("E:/EntroLLM/NHANES 2005-2006/paxraw_d.xpt") %>% 
  filter(PAXSTAT==1 & PAXCAL==1) %>% 
  drop_na(PAXINTEN) %>% 
  select(SEQN,PAXN,PAXINTEN) 

# remove subjects with missing values and average the physical activity data over 5-minute intervals for each subject

missing1=paxraw_d1 %>% 
  group_by(SEQN) %>%
  summarise(n=n()) %>%
  filter(n !=24*60*7)

data1 <- paxraw_d1 %>% 
  mutate(min5 = floor((PAXN - 1) / 5) + 1) %>%   # add min5 variable，ranges from 1 to 2016
  group_by(SEQN, min5) %>%
  summarise(intensity = mean(PAXINTEN)) %>% 
  anti_join(missing1, by = "SEQN") %>% 
  inner_join(bmx_d1,by="SEQN") %>% 
  inner_join(demo_d1,by="SEQN") # 3486 subjects


missing2=paxraw_d2 %>% 
  group_by(SEQN) %>%
  summarise(n=n()) %>%
  filter(n !=24*60*7)

data2 <- paxraw_d2 %>% 
  mutate(min5 = floor((PAXN - 1) / 5) + 1) %>%   # add min5 variable，ranges from 1 to 2016
  group_by(SEQN, min5) %>%
  summarise(intensity = mean(PAXINTEN)) %>% 
  anti_join(missing2, by = "SEQN") %>% 
  inner_join(bmx_d2,by="SEQN") %>% 
  inner_join(demo_d2,by="SEQN") # 3457 subjects

# a total of 6943 subjects
data_6943subjects=rbind(data1,data2)%>% 
  rename(Time=min5,Gender=RIAGENDR,Age=RIDAGEYR,Race=RIDRETH1,Education=DMDEDUC2,Married=DMDMARTL,PIR=INDFMPIR) %>%
  dplyr::select(SEQN,BMXBMI,Time,Gender,Age,Race,Education,Married,PIR,intensity) 


# recode variables

# BMI - 0: <25, 1:>=25(overweight & obesity)
# Gender - 1:male, 2:female
# Race - 1:Mexican American, 2:Other Hispanic, 3:Non-Hispanic White, 4:Non-Hispanic Black, 5:Other Race(including Multi-Racial)
# Education - 1:Less Than 9th Grade, 2:9-11th Grade (Includes 12th grade with no diploma), 3:High School Grad/GED or Equivalent, 4:Some College or AA degree, 5:College Graduate or above
# Married - 1	Married, 2:Widowed, 3:Divorced, 4:Separated, 5:Never married, 6:Living with partner
# PA(physical activity) - 0:intensity<100 counts/min(sedentary), 1:100<=intensity<760 counts/min(light),2:760<=intensity<2200 counts/min(lifestyle), 3:2200<=intensity<6000 counts/min(moderate), 4:intensity>=6000 counts/min(vigorous)

data_wide = data_6943subjects %>% 
  mutate(BMI=ifelse(BMXBMI>=25, 1, 0),PA=ifelse(intensity<100,0, ifelse(intensity<760,1,ifelse(intensity<2200,2,ifelse(intensity<6000,3,4))))) %>%
  select(-BMXBMI,-intensity) %>% 
  pivot_wider(names_from = Time, values_from = PA, names_prefix = "Time")

save(data_wide, file="E:/EntroLLM/data_wide.RData")
write.csv(data_wide, file="E:/EntroLLM/data_wide.csv")
