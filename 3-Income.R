# === 1. Подключение библиотек ===

library(ggplot2)     # Для визуализации
library(caret)       # Для разделения данных и оценки моделей
library(MASS)        # Для линейного дискриминантного анализа (LDA)
library(dplyr)       # Для обработки данных
library(pROC)        # Для построения ROC-кривых и оценки качества моделей


# === 2. Загрузка и предварительная обработка данных ===

# Загрузка данных из CSV файла; разделитель — точка с запятой
census_income = read.csv("/home/census_income.csv", sep = ";", stringsAsFactors = FALSE)

# Замена всех значений " ?" на NA
census_income[census_income == " ?"] = NA 

# Удаление строк с пропущенными значениями
data_clean = na.omit(census_income)

# Преобразование всех строковых переменных в факторы
data_clean = data_clean %>% mutate(across(where(is.character), as.factor))

# Проверка структуры очищенных данных
str(data_clean)


# === 3. Визуальный анализ зависимости уровня дохода от пола ===

# Группированная столбчатая диаграмма: доли уровней дохода по полу
ggplot(data_clean, aes(x = sex, fill = income)) +
  geom_bar(position = "fill") +
  labs(title = "Количество людей с разным уровнем дохода по полу",
       x = "Пол",
       y = "Количество",
       fill = "Доход") +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_fill_manual(values = c(" <=50K." = "green", " >50K." = "purple")) +
  theme_minimal()


# === 4. Нормализация количественных переменных ===

# Определение числовых переменных
numeric_vars = sapply(data_clean, is.numeric)

# Min-Max нормализация (в диапазон [0,1])
min_max_norm = function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Применение нормализации ко всем числовым переменным
data_normalized = data_clean %>%
  mutate(across(which(numeric_vars), min_max_norm))

# Просмотр первых строк нормализованных данных
head(data_normalized)


# === 5. Разделение выборки на обучающую и тестовую ===

# Приведение переменной дохода к бинарному фактору с уровнями
data_normalized$income = factor(data_normalized$income, levels = c(" <=50K.", " >50K."))

# Установка случайного зерна для воспроизводимости
set.seed(123)

# Создание индексов для обучающей выборки (70%)
train_index = createDataPartition(data_normalized$income, p = 0.7, list = FALSE)

# Формирование обучающей и тестовой выборок
train_data = data_normalized[train_index, ]
test_data = data_normalized[-train_index, ]

# Проверка размеров
cat("Размер обучающей выборки:", nrow(train_data), "\n")
cat("Размер тестовой выборки:", nrow(test_data), "\n")


# === 6. Обучение моделей: логистическая регрессия и LDA ===

# Повторное явное указание уровней для целевой переменной
train_data$income = factor(train_data$income, levels = c(" <=50K.", " >50K."))

# Обучение логистической регрессии
log_model <- glm(income ~ age + workclass + education + sex + hours.per.week + occupation + capital.gain + capital.loss,
                 data = train_data, family = binomial)

# Вывод параметров логистической модели
summary(log_model)

# Обучение модели LDA
lda_model <- lda(income ~ age + workclass + education + sex + hours.per.week + occupation + capital.gain + capital.loss,
                 data = train_data)

# Вывод параметров модели LDA
print(lda_model)


# === 7. Оценка моделей на тестовой выборке ===

# --- Логистическая регрессия ---

# Предсказания вероятностей
log_pred_probs = predict(log_model, newdata = test_data, type = "response")

# Классификация по порогу 0.5
log_pred_classes = ifelse(log_pred_probs > 0.5, " >50K.", " <=50K.")

# Матрица ошибок
log_conf_matrix = table(Predicted = log_pred_classes, Actual = test_data$income)
log_conf_matrix

# Расчет метрик
log_accuracy <- sum(diag(log_conf_matrix)) / sum(log_conf_matrix)
log_precision <- log_conf_matrix[2, 2] / sum(log_conf_matrix[2, ])
log_specificity <- log_conf_matrix[1, 1] / sum(log_conf_matrix[1, ])
log_sensitivity <- log_conf_matrix[2, 2] / sum(log_conf_matrix[, 2])
log_f1 <- 2 * (log_precision * log_sensitivity) / (log_precision + log_sensitivity)

# Построение ROC-кривой
log_roc <- roc(test_data$income, log_pred_probs, auc=TRUE)
plot(log_roc, main = "ROC Curve - Logistic Regression", grid.col = c("black", "black"), 
     grid = c(0.1, 0.1),
     print.thres.col='darkblue',
     print.thres.cex=0.95, 
     print.auc = TRUE,
     print.thres = TRUE, col="red")
abline(a = 0, b = 1, lty = 2)

# Заливка под кривой
polygon(c(0, log_roc$specificities, 1), 
        c(0, log_roc$sensitivities, 0),
        col = rgb(0.5, 0.5, 0.5, 0.2),         
        border = NA)

# Вывод метрик логистической модели
cat("Logistic Regression Metrics:\n")
cat("Accuracy:", log_accuracy, "\n")
cat("Sensitivity:", log_sensitivity, "\n")
cat("Specificity:", log_specificity, "\n")
cat("Precision:", log_precision, "\n")
cat("F1 Score:", log_f1, "\n")


# --- Линейный дискриминантный анализ (LDA) ---

# Предсказания классов
lda_pred <- predict(lda_model, newdata = test_data)
lda_pred_classes <- lda_pred$class

# Матрица ошибок
lda_conf_matrix <- table(Predicted = lda_pred_classes, Actual = test_data$income)
lda_conf_matrix

# Расчет метрик
lda_accuracy <- sum(diag(lda_conf_matrix)) / sum(lda_conf_matrix)
lda_precision <- lda_conf_matrix[2, 2] / sum(lda_conf_matrix[2, ])
lda_specificity <- lda_conf_matrix[1, 1] / sum(lda_conf_matrix[1, ])
lda_sensitivity <- lda_conf_matrix[2, 2] / sum(lda_conf_matrix[, 2])
lda_f1 <- 2 * (lda_precision * lda_sensitivity) / (lda_precision + lda_sensitivity)

# Построение ROC-кривой
lda_roc <- roc(test_data$income, lda_pred$posterior[, 2])
plot(lda_roc, main = "ROC Curve - LDA", grid.col = c("black", "black"), 
     grid = c(0.1, 0.1),
     print.thres.col='darkblue',
     print.thres.cex=0.95, 
     print.auc = TRUE,
     print.thres = TRUE, col="red")
abline(a = 0, b = 1, lty = 2)

# Заливка под кривой
polygon(c(0, lda_roc$specificities, 1), 
        c(0, lda_roc$sensitivities, 0),
        col = rgb(0.5, 0.5, 0.5, 0.2),         
        border = NA)

# Вывод метрик LDA модели
cat("\nLDA Metrics:\n")
cat("Accuracy:", lda_accuracy, "\n")
cat("Sensitivity:", lda_sensitivity, "\n")
cat("Specificity:", lda_specificity, "\n")
cat("Precision:", lda_precision, "\n")
cat("F1 Score:", lda_f1, "\n")

