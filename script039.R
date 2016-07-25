Sys.setlocale('LC_ALL','utf-8')
library(MASS)
library(quantreg)
library(caret)
library(brnn)
library(doMC)
registerDoMC(cores = 4)
source('funs.R', encoding = 'UTF-8')

###############################################################
#
# ансамбль из 500 переобученных моделей на конфигурацию
# используются сколнные к переобучению варианты и нейронки
# 
###############################################################


#####################
# Загрузка-предобработка
#####################
load('datasets.RData')
# очень неприятные наблюдения
drop.rows <- c(2623, 3458, 3505, 180, 3311, 3313, 3937, 3945, 1863, 4256)
X.train <- X.train[!(as.numeric(rownames(X.train)) %in% drop.rows),]


#####################
# обучение стека
#####################
models <- list()
# количество переобученных моделей
repeats <- 500
# для воспроизводимости
seeds <- 1:repeats + 100
# соотношение на обучающую/отложенную выборку
p.train <- 3/4

##########################################
# проход по конфигурациям систем
##########################################
for (i in 1:length(confs)){
    cat(i, ' . ')
    X.conf <- subset(X.train, conf==confs[i])
    X.conf$conf <- NULL
    X.conf <- X.conf[!(as.numeric(rownames(X.conf)) %in% drop.rows),]

    ##########################################
    # выбираем лучшие 500 (repeats) моделей
    ##########################################
    models.conf <- list()
    for (j in 1:repeats){
        set.seed(seeds[j])
        in.train <- sample(1:nrow(X.conf), p.train*nrow(X.conf))
        X.conf.train <- X.conf[in.train,]
        X.conf.test <- X.conf[-in.train,]
        # и выбираем лучшую, переобученную модель
        fit <- lm(time ~ ., data=X.conf.train)
        mape.best <- mape(X.conf.test$time, predict(fit, X.conf.test))
        
        # rlm base
        fit.new <- rlm(time~., data=X.conf.train)
        mape.new <- mape(X.conf.test$time, predict(fit.new, X.conf.test))
        if (mape.new < mape.best){
            fit <- fit.new
            mape.best <- mape.new
        }
        
        # rq base
        fit.new <- rq(time~., data=X.conf.train)
        mape.new <- mape(X.conf.test$time, predict(fit.new, X.conf.test))
        if (mape.new < mape.best){
            fit <- fit.new
            mape.best <- mape.new
        }
        
        # rq.tau.best(X.conf.train, 'time~mkn+I(k/m)+I(n/k)+I(n/m)')
        fit.new <- rq.tau.best(X.conf.train, 'time~mkn+I(k/m)+I(n/k)+I(n/m)')
        mape.new <- mape(X.conf.test$time, predict(fit.new, X.conf.test))
        if (mape.new < mape.best){
            fit <- fit.new
            mape.best <- mape.new
        }
        
        # rlm(X.conf.train, 'time~mkn+I(k/m)+I(n/k)+I(n/m)')
        fit.new <- rlm(time~mkn+I(k/m)+I(n/k)+I(n/m), X.conf.train)
        mape.new <- mape(X.conf.test$time, predict(fit.new, X.conf.test))
        if (mape.new < mape.best){
            fit <- fit.new
            mape.best <- mape.new
        }
        
        # rq(.*I(mkn>medinan(mkn)))
        thr.med <- median(X.conf$mkn)
        fit.new <- rq(time~.+mkn*I(mkn>thr.med), data=X.conf.train)
        mape.new <- mape(X.conf$time, predict(fit.new, X.conf.test))
        if (mape.new < mape.best){
            fit <- fit.new
            mape.best <- mape.new
        }
        
        # rlm(.*I(mkn>medinan(mkn)))
        thr.med <- median(X.conf$mkn)
        fit.new <- rlm(time~.+mkn*I(mkn>thr.med), data=X.conf.train)
        mape.new <- mape(X.conf$time, predict(fit.new, X.conf.test))
        if (mape.new < mape.best){
            fit <- fit.new
            mape.best <- mape.new
        }
        
        # нейронная сеть с байесовской регуляризацией
        {
            f = file();sink(file=f) # глушилка 
            # fit.new <- brnn(time~., data=X.conf.train, neurons=7)
            # если метод не сходится, отлавливаем это
            fit.new <- tryCatch(brnn(time~., data=X.conf.train, neurons=7), 
                                error=function(e) fit)
            sink(); close(f)
            mape.new <- mape(X.conf$time, predict(fit.new, X.conf.test))
            if (mape.new < mape.best){
                fit <- fit.new
                mape.best <- mape.new
            }
        }
        
        # добавляем модель в ансамбль
        models.conf[[j]] <- fit
    }
    
    # добавляем ансамбль в стек
    models[[confs[i]]] <- models.conf
}



#####################
# предсказания
#####################
# подготовим X.test, y.test
y.test <- rep(1, nrow(X.test))
y.train <- rep(1, nrow(X.train))
X.test <- dplyr::inner_join(X.test, X.uni, by=colnames(X.uni)[2:ncol(X.uni)])[cols]
# int -> numeric
X.test[,1:3] <- lapply(X.test[,1:3], as.numeric)
X.test$mkn <- X.test$m * X.test$k * X.test$n

# предсказываем y.test
for (i in 1:length(confs)){
    cat(i, '. ')
    
    # работаем с каждой конфигурацией отдельно
    X.conf <- subset(X.test, conf==confs[i])
    y.pred <- rep(0, nrow(X.conf))
    
    # считаем одну конфигурацию для стопки моделей
    models.conf <- models[[confs[i]]]
    for (j in 1:repeats){
        # вес каждой модели -- 1/repeats
        y.pred <- y.pred + predict(models.conf[[j]], X.conf) / repeats
    }
    
    # предсказания для текущей конфигурации
    pred.idx <- as.numeric(rownames(X.conf))
    y.test[pred.idx] <- y.pred
}


# здесь катофф трик без вычислений на обучаеющей выборке
cutoff.low <- 1.041
# всего надо сделать замен:
sum(y.test<cutoff.low)
# заменяем маленькие величины в y.test
y.test.origin <- y.test
y.test[y.test<cutoff.low] <- cutoff.low

# записываем результаты в файл, repeats=500
out <- data.frame(time = y.test)
write.table(out, file = 'subs/sub039.csv', sep=",",  col.names=FALSE, row.names=FALSE)
# сохраним стек моделей
saveRDS(models, 'models/model039.RDS')
# m039 <- readRDS('models/model039.RDS')


