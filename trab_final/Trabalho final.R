#Trabalho Final
#Diego Leonel Costa

# CArregando os pacotes
pacman::p_load(crosstalk, dplyr, DT, plotly, ade4, car, mboost, caret, corrplot, data.table, dplyr, forcats, funModeling, ggplot2, mlbench, mltools, randomForest, rattle, tidyverse, DALEX, doParallel)

#Escolhendo a base

municipios_pe <- read.csv2("bases_originais/clusters_municipios_pe.csv", fileEncoding = "latin1")
municipios_pe_f <- municipios_pe %>% 
  select("pop", "pib", "vab", "icms", "ipi", "ipva", "salario_medio", "pop_ocu_per")                         


# Dividindo os dados treinamento e teste (70% treinamento, 30% teste)

treino_MUNICIPIOS <- createDataPartition(municipios_pe_f$salario_medio, p = 0.7, list = FALSE)
dados_treino <- municipios_pe_f[treino_MUNICIPIOS, ]
dados_teste <- municipios_pe_f[-treino_MUNICIPIOS, ]

# Crie um modelo de regressão linear
modelo <- lm(salario_medio ~., data=dados_treino)

# Faça previsões no conjunto de teste
previsoes <- predict(modelo, newdata = dados_teste)

# Avalie o desempenho do modelo
resultado <- data.frame(observado = dados_teste$salario_medio, previsao = previsoes)
print(resultado)

# Calcule métricas de desempenho (por exemplo, MSE - Erro Quadrático Médio)
mse <- mean((resultado$observado - resultado$previsao)^2)
print(paste("Erro Quadrático Médio (MSE):", mse))

##############
# Treino e Teste: Pré-processamento
# Controle de treinamento
train.control <- trainControl(method = "cv", number = 10, verboseIter = T) # controle de treino
#########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
########O PIB TEM QUE SER A VARIÁVEL DE SAIDA!########
#######
sal_munic <- salario_medio ~ pop + pib + vab + icms + ipi + ipva + pop_ocu_per

registerDoParallel(cores = detectCores() - 1)

# Treinamentos
## Regressão Linear penalizada
municipios_LM <- train(
  sal_munic, 
  data = dados_treino, 
  method = "glmnet", 
  trControl = train.control, tuneLength = 20)

plot(municipios_LM)
summary(municipios_LM) # sumário do modelo linear
plot(varImp(municipios_LM))

coeficientes <- coef(municipios_LM$finalModel, municipios_LM$bestTune$lambda)
coeficientes

## Árvore de Decisão
municipios_RPART <- train(
  sal_munic, 
  data = dados_treino, 
  method = "rpart", 
  trControl = train.control, tuneLength = 20)

plot(municipios_RPART)
summary(municipios_RPART)
fancyRpartPlot(municipios_RPART$finalModel) # desenho da árvore
plot(varImp(municipios_RPART)) # importância das variáveis

# Bagging com Floresta Aleatória
municipios_RF <- train(
  sal_munic, 
  data = dados_treino, 
  method = "cforest", 
  trControl = train.control, tuneLength = 20)

plot(municipios_RF) # evolução do modelo
plot(varImp(municipios_RF)) # plot de importância


# Boosting com Boosted Generalized Linear Model
municipios_GLMB <- train(
  sal_munic, 
  data = dados_treino, 
  method = "glmboost", 
  trControl = train.control, tuneLength = 20)

plot(municipios_GLMB) # evolução do modelo
print(municipios_GLMB) # modelo
summary(municipios_GLMB) # sumário

#Analisando qual o melhor modelo
melhor_modelo <- resamples(list(LM = municipios_LM, RPART = municipios_RPART, RF = municipios_RF, GLMBOOST = municipios_GLMB))
melhor_modelo

summary(melhor_modelo)
#RF ganhou

#testando o modelo
predVals <- caret::extractPrediction(municipios_RF, testX = dados_teste[ , -7]) #deu erro

pred_modelos <- data.frame(
  obs = dados_teste$salario_medio,
  rf = predict(municipios_RF, dados_teste)
) %>% mutate (rf_res = obs - rf)

ggplot(pred_modelos, aes(obs, rf)) + 
  geom_point() + geom_smooth()

ggplot(pred_modelos, aes(rf, rf_res)) + 
  geom_point() + geom_hline(yintercept = 0, color = "red")

####################




  