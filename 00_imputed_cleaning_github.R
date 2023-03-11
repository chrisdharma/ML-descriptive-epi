#load data
library(tableone)
library(naniar)
library(ggplot2)
library(tidyr)
library(mice)
library(corrplot)
library(dplyr)
library(miceadds)

#Select only those who completed baseline survey
dat<- dat[dat$baseline_complete==2,]

#recode those with at least once a week use as frequent
dat$alcohol <- recode(dat$alcohol,'2'=0,'1'=1,'0'=0)
dat$cannabis <- recode(dat$cannabis,'2'=0,'1'=1,'0'=0)

#Reducing the number of categories
dat$ind_income <- recode(dat$ind_income,'1'=1,'2'=2,'3'=3,'4'=4,'5'=5,'6'=5)
dat$curr_orient2 <- recode(dat$curr_orient2,'0'=0,'1'=1,'2'=2,'3'=3,'4'=4,'5'=5,'6'=6,'7'=1,'8'=6)

dat_vars <- dat %>% select(c(suicidal,age,province,curr_orient2,gender,trans,intersex,
                             poc,residence,education,house_income,ind_income,rural_city,where_live,
                             gen_health,fitness1,mental_health,stresslife,diagnosis,
                             con_eating,con_anxiety,con_ADD,con_ADHD,con_depression,treat_comorbid,
                             con_OCD,con_panic,con_PTSD,con_others,con_bipolar,receive_help,
                             cigs_smoked,curr_smoke,use_cigar,use_wp,use_smokeless,covid,
                             plan_quit,quit_attempts,tailored,quit_support,time_vape,curr_vape,alcohol,disability,
                             alcohol_amount,cannabis,poppers,crystal_meth,crack,cocaine,heroin,pres_opioids,psychedelics,
                             fentanyl,GHB,tranquilizers,drug_others,substances_covid,seek_help,central,not_sig,imprtant,
                             understand,mom,dad,sibs,partner,ext_fam,new_frnd,old_frnd,co_work,employr,relig_mem,stranger,
                             famdoc,oth_hlth,classmt,teach,part_q,pos_q,bond_q,proud_q,polit_q,solv_q,prob_q,norm_q,
                             pass_q,relat_q,hit_q,police_q,live_q,job_q,names_q,asslt_q,frnd_q,
                             hurt_q,fam_q,relig_q,comfrt,control,public,change,seen,sustain,anonym,promisc,
                             nervous,public_plc,bars,advnce,mentalill,drinker,streetdrug,jail,divorce,slap,feel1,feel2,feel3,feel4,feel5,feel6,feel7,
                             beat,swear,inapptouch,inapptouchyou,forced,employ))

summary(dat_vars)

dat_vars[sapply(dat_vars, is.nan)] <- NA

lapply(dat_vars,function(i) {
  table(i, useNA="ifany")})

#check for missing data
sapply(dat_vars, function(x) sum(is.na(x)))

vis_miss(dat_vars,sort_miss=TRUE) 

q.hd <- dat_vars %>% summarise_all(~sum(is.na(.)))
q2.hd <- t(q.hd)
q3.hd <- data.frame('Number_of_missingness'= q2.hd[,1],
                    'percent_missingness'=round(q2.hd[,1]/nrow(dat_vars)*100, digit=2))
q3.hd <- q3.hd[order(q3.hd$percent_missingness,q3.hd$Number_of_missingness),]

m.hd <- q3.hd[q3.hd$percent_missingness>=5,]
dim(m.hd)

#change factors
names<-c("province","curr_orient2","gender","trans","intersex","treat_comorbid","disability",
         "poc","residence","education","house_income","ind_income","rural_city",
         "where_live","fitness1","diagnosis","alcohol","alcohol_amount","cannabis","poppers","crystal_meth","crack","cocaine","heroin","pres_opioids",
         "fentanyl","GHB","tranquilizers","drug_others","psychedelics",
         "con_eating","con_anxiety","con_ADD","con_ADHD","con_depression","con_bipolar",
         "con_OCD","con_panic","con_PTSD","con_others","use_cigar","use_wp","use_smokeless","covid",
         "plan_quit","quit_attempts","tailored","quit_support","time_vape",
         "curr_vape","substances_covid","seek_help","employ")
dat_vars[,names]<-lapply(dat_vars[,names],factor)

names2<-c("mentalill","drinker","streetdrug","jail","divorce","slap","beat","swear",
          "inapptouch","inapptouchyou","forced")

dat_vars[,names2]<-sapply(dat_vars[,names2],as.numeric)

##Imputation for missing data. 
dat1.hd <- dat_vars


init <- mice(dat1.hd, maxit=0) 
meth <- init$method
predM <- init$predictorMatrix

predM[,"alcohol"] <- 0

set.seed(123)
imputed.hd <- mice::mice(dat1.hd, method='pmm', predictorMatrix=predM, m=5)
#pmm = imputation for any types of variables, rather than LDA
summary(imputed.hd)

# select the first copy
df1 <- complete(imputed.hd,1)
#We're just doing it with one imputation for illustration purposes

#Create function to scale the data on the data frames
scalevars <- function(df) {
  sapply(df, function(x) sum(is.na(x)))
  df <- df %>% mutate(cen_identity = select(.,central:understand) %>% rowSums(na.rm=TRUE))
  df <- df %>% mutate(outness = select(.,mom:teach) %>% rowMeans(na.rm=TRUE)) 
  df <- df %>% mutate(connect_com = select(.,part_q: prob_q) %>% rowSums(na.rm=TRUE)) 
  df <- df %>% mutate(ace = select(.,mentalill: forced) %>% rowSums(na.rm=TRUE)) 
  df <- df %>% mutate(phobia = select(.,comfrt: advnce) %>% rowSums(na.rm=TRUE))
  df <- df %>% mutate(per_stigma = select(.,c(norm_q:relat_q,hurt_q,fam_q)) %>% rowSums(na.rm=TRUE))
  df <- df %>% mutate(en_stigma = select(.,c(hit_q: frnd_q,relig_q)) %>% rowSums(na.rm=TRUE)) 
  #Need to sub CES-D scores as well
  df <- df %>% mutate(cesd_score = select(.,c(feel1: feel7)) %>% rowSums(na.rm=TRUE)) 
  # df <- df %>% mutate(per_sm_risk = select(.,c(risk1: risk4)) %>% rowSums(na.rm=TRUE)) 
  # df <- df %>% mutate(per_sm_lgbtq = select(.,c(accept: culture)) %>% rowSums(na.rm=TRUE)) 
  df[,c("cen_identity", "outness", "connect_com", "ace", "phobia", "per_stigma","en_stigma","cesd_score")] <- 
    df %>% select(cen_identity, outness, connect_com, ace, phobia, per_stigma, en_stigma, cesd_score) %>% scale()
  df <- df %>% select(-c(central:understand,mom:teach,part_q:prob_q,mentalill:forced,
                         comfrt: advnce,norm_q:relat_q,hurt_q,fam_q,hit_q: frnd_q,relig_q, feel1:feel7))
  return(df)
}

df1 <- df1 %>% mutate(alcohol_abuse = ifelse(alcohol == 1 & alcohol_amount == 1,1,0),
                      alcohol_abusec = ifelse(alcohol == 1 & alcohol_amount == 1,"class1","class0")) %>%
              select(-c(alcohol,alcohol_amount,substances_covid,tailored,quit_attempts,quit_support,time_vape))

df1<-scalevars(df1)
table(df1$alcohol_abuse)
table(df1$alcohol_abusec)

save(df1,file="impute_Table2.Rdata")
