library(arules)
library(arulesViz)
library(tidyverse)
dataset = read.csv("./data/BlackFriday.csv")
customers_products = dataset %>%
                        select(User_ID, Product_ID) %>%   
                        group_by(User_ID) %>%                       
                        arrange(User_ID) %>%               
                        mutate(id = row_number()) %>%     
                        spread(User_ID, Product_ID) %>%   
                        t()                               
customers_products = customers_products[-1,]
write.csv(customers_products, file = 'customers_products.csv')
customersProducts = read.transactions('customers_products.csv', sep = ',', rm.duplicates = TRUE) 
rules = apriori(data = customersProducts,parameter = list(support = 0.04, confidence = 0.40, maxtime = 0))
inter_rules=subset(rules,subset=lift>=3.0)
plot(inter_rules,method='graph')