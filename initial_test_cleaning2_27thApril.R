library(dplyr)

entire_data = read.csv(file = "D:\\Research papers/PhD Spring 2020 Coursework/TermProjects/NLP-term-project/corpus_analysis_26thApril.csv", header = TRUE, stringsAsFactors = FALSE)

entire_data$translated_ascii = iconv(entire_data$translated, to='ascii', sub = '')
entire_data = entire_data %>% mutate(trans_ascii_len = nchar(translated_ascii))
entire_data = entire_data %>% mutate(trans_boolean = if_else(trans_ascii_len > 150, 1, 0))

entire_data1 = subset(entire_data, trans_boolean == 1) %>% select(-trans_boolean)

# We scale the timestamp information 
min_time_val = min(entire_data$timestamp)
# Epoch to date : Thursday, February 7, 2019 10:43:28 PM GMT+05:30
max_time_val = max(entire_data$timestamp)
# Epoch to date ; Friday, February 8, 2019 4:49:21 PM GMT+05:30
entire_data2 = entire_data %>% mutate(timestamp = (timestamp - min_time_val)/1000)
entire_data3 = entire_data2 %>% select(group_id_anonymized, lang, timestamp)

entire_grps = entire_data2 %>% group_by(group_id_anonymized) %>% summarize(total_posts = n(), start_time = min(timestamp), last_time = max(timestamp)) %>% arrange(desc(total_posts))

entire_grps_time = entire_grps %>% arrange(start_time, last_time)

#----------------------- Inter-arrival time (IAT) analysis starts ---------------------------------------------


# Starting with the Inter-arrival time analysis
# We consider only those groups for which we have at least 5 posts
entire_grps_iat = subset(entire_grps_time, total_posts >= 5)

# Reduces to 53 Whatsapp groups. Since, the time duration is less than one day, it is good threshold for stating that the Whatsapp group is sufficiently active
# Only 21.2% of the groups are consider active based on our assumption 

iat_authids = entire_grps_iat %>% select(group_id_anonymized)
entire_data_iat = merge(iat_authids, entire_data3, by='group_id_anonymized')
# Leads to a total of 648 posts out of 1000

entire_data_iat = entire_data_iat %>% arrange(group_id_anonymized, timestamp)


iat_in_sec = NULL

for (i in 1:nrow(entire_data_iat)) {
  if(i == 1){
    iat_in_sec = c(iat_in_sec, 0)
  }
  else if(entire_data_iat[i,1] != entire_data_iat[i-1, 1]){
    iat_in_sec = c(iat_in_sec, 0)
  }
  else{
    iat_in_sec = c(iat_in_sec, entire_data_iat[i,3] - entire_data_iat[i-1,3])
  }
}
entire_data_iat$iat_in_sec = iat_in_sec

entire_data_iat1 = subset(entire_data_iat, iat_in_sec > 0)
x = entire_data_iat1$timestamp

library(ggplot2)
plot(sort(x) , 1-ecdf(x)(sort(x) ), main= "CCDF plot of inter-arrival time of posts", xlab="Inter-arrival time in seconds", ylab="CCDF")

# Inter-arrival times follow a heavy-tailed distribution which is a common posting behavior by social media users

#----------------- LIWC analysis starts -------------------------------------------

# We perform LIWC analysis on only the english posts or those are translated to English and ocntain more than 150 characters

entire_data1_liwc = entire_data1 %>% select(group_id_anonymized, timestamp, translated_ascii)
entire_data1_liwc$seq_id = seq(1,nrow(entire_data1_liwc))
write.csv(entire_data1_liwc, file = "D:\\Research papers/PhD Spring 2020 Coursework/TermProjects/NLP-term-project/fear_speech_liwc_analysis_28thApril.csv", row.names = FALSE)

# Raeding the Liwc output file
entire_data_liwc_output = read.csv(file = "D:\\Research papers/PhD Spring 2020 Coursework/TermProjects/NLP-term-project/fear_speech_liwc_output_28thApril.csv", header = TRUE, stringsAsFactors = FALSE)
entire_data_liwc_output$seq_id = seq(1, nrow(entire_data_liwc_output))


# Summary info of the LIWC file
str1 = '
 Ppron       Inhib             Space       Filler            Ipron        Time      Percept       
 Min.   :0   Min.   :0.00000   Min.   :0   Min.   :0.00000   Min.   :0   Min.   :0   Min.   :0.00000  
 1st Qu.:0   1st Qu.:0.00000   1st Qu.:0   1st Qu.:0.00000   1st Qu.:0   1st Qu.:0   1st Qu.:0.00000  
 Median :0   Median :0.00000   Median :0   Median :0.00000   Median :0   Median :0   Median :0.00000  
 Mean   :0   Mean   :0.01582   Mean   :0   Mean   :0.00703   Mean   :0   Mean   :0   Mean   :0.05097  
 3rd Qu.:0   3rd Qu.:0.00000   3rd Qu.:0   3rd Qu.:0.00000   3rd Qu.:0   3rd Qu.:0   3rd Qu.:0.00000  
 Max.   :0   Max.   :2.00000   Max.   :0   Max.   :1.00000   Max.   :0   Max.   :0   Max.   :8.00000  
     Verbs              Quant            Discrep     Relativ           Affect             You        Cause  
 Min.   :0.000000   Min.   :0.00000   Min.   :0   Min.   : 0.000   Min.   :0.00000   Min.   :0   Min.   :0  
 1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0   1st Qu.: 2.000   1st Qu.:0.00000   1st Qu.:0   1st Qu.:0  
 Median :0.000000   Median :0.00000   Median :0   Median : 3.000   Median :0.00000   Median :0   Median :0  
 Mean   :0.008787   Mean   :0.09666   Mean   :0   Mean   : 5.223   Mean   :0.01054   Mean   :0   Mean   :0  
 3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0   3rd Qu.: 5.000   3rd Qu.:0.00000   3rd Qu.:0   3rd Qu.:0  
 Max.   :1.000000   Max.   :3.00000   Max.   :0   Max.   :77.000   Max.   :1.00000   Max.   :0   Max.   :0  
      Prep       Relig              Body        Bio                We        Assent             Incl  
 Min.   :0   Min.   : 0.0000   Min.   :0   Min.   : 0.0000   Min.   :0   Min.   :0.00000   Min.   :0  
 1st Qu.:0   1st Qu.: 0.0000   1st Qu.:0   1st Qu.: 0.0000   1st Qu.:0   1st Qu.:0.00000   1st Qu.:0  
 Median :0   Median : 0.0000   Median :0   Median : 0.0000   Median :0   Median :0.00000   Median :0  
 Mean   :0   Mean   : 0.8155   Mean   :0   Mean   : 0.3638   Mean   :0   Mean   :0.02109   Mean   :0  
 3rd Qu.:0   3rd Qu.: 0.0000   3rd Qu.:0   3rd Qu.: 0.0000   3rd Qu.:0   3rd Qu.:0.00000   3rd Qu.:0  
 Max.   :0   Max.   :46.0000   Max.   :0   Max.   :18.0000   Max.   :0   Max.   :3.00000   Max.   :0  
    Leisure            AuxVb        Hear             They       Posemo          Article       Excl  
 Min.   : 0.0000   Min.   :0   Min.   :0.0000   Min.   :0   Min.   : 0.000   Min.   :0   Min.   :0  
 1st Qu.: 0.0000   1st Qu.:0   1st Qu.:0.0000   1st Qu.:0   1st Qu.: 0.000   1st Qu.:0   1st Qu.:0  
 Median : 0.0000   Median :0   Median :0.0000   Median :0   Median : 1.000   Median :0   Median :0  
 Mean   : 0.5606   Mean   :0   Mean   :0.1845   Mean   :0   Mean   : 1.694   Mean   :0   Mean   :0  
 3rd Qu.: 0.0000   3rd Qu.:0   3rd Qu.:0.0000   3rd Qu.:0   3rd Qu.: 2.000   3rd Qu.:0   3rd Qu.:0  
 Max.   :16.0000   Max.   :0   Max.   :5.0000   Max.   :0   Max.   :35.000   Max.   :0   Max.   :0  
      Home            Friends     Present           Numbers     CogMech             I          Work       
 Min.   : 0.0000   Min.   :0   Min.   : 0.0000   Min.   :0   Min.   : 0.000   Min.   :0   Min.   : 0.000  
 1st Qu.: 0.0000   1st Qu.:0   1st Qu.: 0.0000   1st Qu.:0   1st Qu.: 0.000   1st Qu.:0   1st Qu.: 0.000  
 Median : 0.0000   Median :0   Median : 0.0000   Median :0   Median : 1.000   Median :0   Median : 1.000  
 Mean   : 0.2794   Mean   :0   Mean   : 0.5431   Mean   :0   Mean   : 1.494   Mean   :0   Mean   : 2.302  
 3rd Qu.: 0.0000   3rd Qu.:0   3rd Qu.: 1.0000   3rd Qu.:0   3rd Qu.: 2.000   3rd Qu.:0   3rd Qu.: 3.000  
 Max.   :11.0000   Max.   :0   Max.   :20.0000   Max.   :0   Max.   :22.000   Max.   :0   Max.   :33.000  
     Tentat           Ingest           Motion           Anger       Achiev           Swear       Death        
 Min.   :0.0000   Min.   : 0.000   Min.   : 0.000   Min.   :0   Min.   : 0.000   Min.   :0   Min.   :0.00000  
 1st Qu.:0.0000   1st Qu.: 0.000   1st Qu.: 0.000   1st Qu.:0   1st Qu.: 0.000   1st Qu.:0   1st Qu.:0.00000  
 Median :0.0000   Median : 0.000   Median : 0.000   Median :0   Median : 1.000   Median :0   Median :0.00000  
 Mean   :0.1283   Mean   : 0.362   Mean   : 1.237   Mean   :0   Mean   : 2.181   Mean   :0   Mean   :0.09139  
 3rd Qu.:0.0000   3rd Qu.: 0.000   3rd Qu.: 2.000   3rd Qu.:0   3rd Qu.: 3.000   3rd Qu.:0   3rd Qu.:0.00000  
 Max.   :6.0000   Max.   :17.000   Max.   :20.000   Max.   :0   Max.   :26.000   Max.   :0   Max.   :3.00000  
     Social           Nonflu             Family           Pronoun      Funct             Feel       
 Min.   : 0.000   Min.   :0.000000   Min.   : 0.0000   Min.   :0   Min.   :  0.00   Min.   :0.0000  
 1st Qu.: 1.000   1st Qu.:0.000000   1st Qu.: 0.0000   1st Qu.:0   1st Qu.:  8.00   1st Qu.:0.0000  
 Median : 2.000   Median :0.000000   Median : 0.0000   Median :0   Median : 16.00   Median :0.0000  
 Mean   : 3.067   Mean   :0.003515   Mean   : 0.4991   Mean   :0   Mean   : 35.78   Mean   :0.2144  
 3rd Qu.: 4.000   3rd Qu.:0.000000   3rd Qu.: 0.0000   3rd Qu.:0   3rd Qu.: 40.00   3rd Qu.:0.0000  
 Max.   :26.000   Max.   :1.000000   Max.   :24.0000   Max.   :0   Max.   :488.00   Max.   :7.0000  
    Certain          Insight           Humans            Sad         Past             See              Future      
 Min.   : 0.000   Min.   : 0.000   Min.   : 0.000   Min.   :0   Min.   : 0.000   Min.   : 0.0000   Min.   : 0.000  
 1st Qu.: 0.000   1st Qu.: 0.000   1st Qu.: 0.000   1st Qu.:0   1st Qu.: 0.000   1st Qu.: 0.0000   1st Qu.: 0.000  
 Median : 1.000   Median : 1.000   Median : 0.000   Median :0   Median : 0.000   Median : 0.0000   Median : 0.000  
 Mean   : 1.192   Mean   : 1.548   Mean   : 1.207   Mean   :0   Mean   : 3.123   Mean   : 0.5255   Mean   : 1.088  
 3rd Qu.: 2.000   3rd Qu.: 2.000   3rd Qu.: 2.000   3rd Qu.:0   3rd Qu.: 3.000   3rd Qu.: 1.0000   3rd Qu.: 1.000  
 Max.   :17.000   Max.   :35.000   Max.   :18.000   Max.   :0   Max.   :70.000   Max.   :10.0000   Max.   :68.000  
    Adverbs           SheHe             Money            Negate           Health             Conj       
 Min.   : 0.000   Min.   : 0.0000   Min.   : 0.000   Min.   : 0.000   Min.   : 0.0000   Min.   : 0.000  
 1st Qu.: 0.000   1st Qu.: 0.0000   1st Qu.: 0.000   1st Qu.: 0.000   1st Qu.: 0.0000   1st Qu.: 1.000  
 Median : 1.000   Median : 0.0000   Median : 0.000   Median : 0.000   Median : 0.0000   Median : 2.000  
 Mean   : 1.967   Mean   : 0.7663   Mean   : 1.641   Mean   : 1.086   Mean   : 0.4148   Mean   : 4.726  
 3rd Qu.: 2.000   3rd Qu.: 0.0000   3rd Qu.: 2.000   3rd Qu.: 1.000   3rd Qu.: 0.0000   3rd Qu.: 5.000  
 Max.   :42.000   Max.   :27.0000   Max.   :48.000   Max.   :20.000   Max.   :16.0000   Max.   :91.000  
      Anx        Negemo          Sexual           seq_id   
 Min.   :0   Min.   : 0.00   Min.   :0.0000   Min.   :  1  
 1st Qu.:0   1st Qu.: 0.00   1st Qu.:0.0000   1st Qu.:143  
 Median :0   Median : 1.00   Median :0.0000   Median :285  
 Mean   :0   Mean   : 1.65   Mean   :0.1336   Mean   :285  
 3rd Qu.:0   3rd Qu.: 2.00   3rd Qu.:0.0000   3rd Qu.:427  
 Max.   :0   Max.   :27.00   Max.   :8.0000   Max.   :569 
'

# We remove columns with min = max = 0
# Columns : Anx, Sad, Pronoun,Swear,Anger,Friends,Numbers,I,Excl,Article,They,AuxVb,Prep,Body,We,Incl,Time,Ppron,Ipron,Space
liwc_output1 = entire_data_liwc_output %>% select(-Anx, -Sad, -Pronoun,-Swear,-Anger,-Friends,-Numbers,-I,-Excl,-Article,-They,-AuxVb,-Prep,-Body,-We,-Incl,-Time,-Ppron,-Ipron,-Space)
# It reduces to 44 columns

# Which LIWC categories are most frequent (persistent) across the corpus
zero_count = NULL
for(j in 1:ncol(liwc_output1)){
  zeroes = 0
  for(i in 1:nrow(liwc_output1)){
    if(liwc_output1[i,j] == 0){
      zeroes = zeroes + 1
    }
  }
  zero_count = c(zero_count, zeroes)
}

output1 = '
> zero_count
 [1] 561 565 555 564 524 569  61 563 569 569 427 463 561 435 504 212 480 415 278 221 517 480 296 196 534 135 567 458
[29]   5 498 253 284 322 285 423 366 231 440 296 324 474 116 249 526   0
> colnames(liwc_output1)
 [1] "Inhib"   "Filler"  "Percept" "Verbs"   "Quant"   "Discrep" "Relativ" "Affect"  "You"     "Cause"   "Relig"  
[12] "Bio"     "Assent"  "Leisure" "Hear"    "Posemo"  "Home"    "Present" "CogMech" "Work"    "Tentat"  "Ingest" 
[23] "Motion"  "Achiev"  "Death"   "Social"  "Nonflu"  "Family"  "Funct"   "Feel"    "Certain" "Insight" "Humans" 
[34] "Past"    "See"     "Future"  "Adverbs" "SheHe"   "Money"   "Negate"  "Health"  "Conj"    "Negemo"  "Sexual" 
[45] "seq_id"
'

# LIWC classes : Few times - Inhib, Filler, Percept, Verbs, Quant, Discrep, Affect, You, Cause, Assent, Hear, Sexual
# Half - NegEmo, Negate, Money, Certain, Insight, Humans, Past
# Always - Relativ, Funct

#-------- Multilingual Language analysis with IAT starts ---------------------

# Testing the language purity of the group. Also, is there any pattern when the language shifts
# Only 6 out of 53 (11.32%) Whatsapp groups (active groups, having more than 5 posts per day) are monolingual

# We now mention monolingual groups with language and id : hi - 530, 3016, 3371; en - 1898, 3339; ml - 4957

# Relation between language shift in group with iat - entire_data_iat
shift_level = NULL
shift_cat = NULL

for(i in 1:nrow(entire_data_iat)){
  if(entire_data_iat[i, 4] == 0){
    shift_level = c(shift_level, -1)
    shift_cat = c(shift_cat, 'None')
  }
  else if(entire_data_iat[i, 2] == entire_data_iat[i-1, 2]){
    shift_level = c(shift_level, -1)
    shift_cat = c(shift_cat, 'None')
  }
  else{
    if(entire_data_iat[i, 4] <= 120){
      shift_level = c(shift_level, 1)
      shift_cat = c(shift_cat, 'First 2 mins')
    }
    else if(entire_data_iat[i, 4] > 120 && entire_data_iat[i, 4] <= 300){
      shift_level = c(shift_level, 2)
      shift_cat = c(shift_cat, '2 to 5 mins')
    }
    else if(entire_data_iat[i, 4] > 300 && entire_data_iat[i, 4] <= 600){
      shift_level = c(shift_level, 3)
      shift_cat = c(shift_cat, '5 to 10 mins')
    }
    else{
      shift_level = c(shift_level, 4)
      shift_cat = c(shift_cat, '> 10 mins')
    }
  }
}

entire_data_iat$shift_level = shift_level
entire_data_iat$shift_cat = shift_cat
entire_data_iat$shift_cat = as.factor(entire_data_iat$shift_cat)

entire_data_lang1 = subset(entire_data_iat, shift_level > 0)
# 157 out of 648 is not first post and has language shift

entire_data_lang1 = entire_data_lang1 %>% mutate(shift_cat = as.factor(shift_level))

entire_data_lang1 = entire_data_lang1 %>% arrange(shift_level)

library(ggplot2)
ggplot(data=entire_data_lang1, aes(x=shift_cat)) + geom_bar(stat="count") + xlab("Inter-arrival time in minutes")

# Observation the language shift increases significantly once it goes over 10 minutes
