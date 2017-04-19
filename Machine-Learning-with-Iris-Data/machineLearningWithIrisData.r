# store current working directory
filepath = getwd()

# define the filename
filename = paste(filepath, "/iris.data", sep="")

# load the CSV file from the local directory
dataset = read.csv(filename)

# set the column names in the dataset
colnames(dataset) <- c("Sepal.Length", "Sepal.width", "Petal.Length", "Petal.width", "Species")

# Print a summary of the dataset
summary(dataset)

# Print view of all data (note: ellipse works nicely here too, instead of pairs):
featurePlot(x = dataset[, 1:4],
	y = dataset$Species
	plot = "pairs"
	#add a key at the top
	auto.key = list(columns = 3))

#create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Species, p = 0.80, list=FALSE)

#select 20% of the data for validation
validation <- dataset[-validation_index,]

#use the remaining 80% of data to train and test the models
dataset <- dataset[validation_index,]

#dimensions of dataset
dim(dataset)

#list types for each attribute
sapplay(dataset, class)

head(dataset)
tail(dataset)

levels(dataset$Species)


percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)

#summarize attribute distributions
summary(dataset)

#split input and output
x <- dataset[,1:4]
y <- dataset[,5]

#boxplay for each attribute on one image
par(mfrow=c(1,4))
for(i in 1:4) {
	boxplot(x[,i], main=names(iris)[i])
}

#barplot for class breakdown
plot(y)

#scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")

#box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")

#density plots for each attribute by class value
scales <-list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

#Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#Linear Discriminant Analysis (determine linear combination of factors)
#LDA
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)

#Classification and Regression Tree analysis (create decision trees based on observed breakpoint conditions)
#CART
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)

#k-Nearest Neighbor (determine k clusters of data for classification)
#kNN
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trcControl=control)

#Support Vector Machine (Create partitioning by hyperplanes)
#SVM
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trcControl=control)

#Random Forest (Aggregate/combination of decision trees)
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trcControl=control)

#Neural Net (using nnet package, which has only a single hidden layer)
set.seed(7)
fit.neural <- train(Species~., data=dataset, method="nnet", metric=metric, trcControl=control, trace=FALSE)

#Look at some interesting internal data foa particular model:
#e.g. functions for fit.neural, additional info about model:
fit.neural$modelInfo
fit.neural$finalModel
 
#compare accuracy of models
dotplot(results)

#summarize best models
print(fit.lda)
print(fit.neural)

#estimate skill of LDA on the validation dataset
predictions <-predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)

#estimate skill of LDA on the validation dataset
predictions2 <-predict(fit.lda, validation)
confusionMatrix(predictions2, validation$Species)

#estimate skill of LDA on the validation dataset
predictions3 <-predict(fit.lda, validation)
confusionMatrix(predictions3, validation$Species)
