dataset = read.csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Position_Salaries.csv')

dataset = dataset[,2:3]

#install.packages("caTools")

#set.seed(123)
#split = sample.split( dataset$Profit, SplitRatio = 0.8 )
#training_set = subset( dataset, split == TRUE  )
#testing_set = subset( dataset, split == FALSE  )

#regression = lm(  formula = Profit ~ ., data = training_set)

#y_predict = predict( regression, newdata =  testing_set )
