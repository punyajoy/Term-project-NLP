# This code creates a cleaner English version of the translated part of the code

library(dplyr)

total_data = read.csv(file = "D:\\Research papers/PhD Spring 2020 Coursework/TermProjects/NLP-term-project/Total_data_annotated.csv", header = TRUE, stringsAsFactors = FALSE)
total_data = total_data %>% select(id, label, translated)

total_data$translated = iconv(total_data$translated, to='ASCII', sub="")
total_data = total_data %>% mutate(eng_len = nchar(translated))
total_data1 = subset(total_data, eng_len > 100)

write.csv(total_data1, file = "D:\\Research papers/PhD Spring 2020 Coursework/TermProjects/NLP-term-project/Total_data_annotated_eng_clean_28thApril.csv", row.names = FALSE)
