# V Cramer's correlation
cv.test <- function(x,y) {
    CV <- sqrt(chisq.test(x, y, correct=FALSE)$statistic /
                  (length(x) * (min(length(unique(x)),length(unique(y))) - 1)))
    print.noquote("Cramer V / Phi:")
    return(as.numeric(CV))
}

# возвращает индексы рядов df идентичных some.row
dup.idx <- function(df, some.row){
    df$n <- 1:nrow(df)
    dplyr::right_join(df, some.row, by=colnames(some.row))$n
}


# считаем MAPE метрику
mape <- function(y, pred){
    mean(abs(y-pred) / y)
}

mape.sum <- function(y, pred){
    sum(abs(y-pred) / y) / length(y)
}

# mape для каждого наблюдения
mape.each <- function(y, pred){
    (y-pred) / y
}

# предсказания для датафрейма X
get.y.pred.m <- function(X, models){
    y.pred <- rep(1, nrow(X))
    for (i in 1:length(confs)){
        X.conf <- subset(X, conf==confs[i])
        # prediction for current model
        pred <- predict(models[[confs[i]]], newdata = X.conf)
        # pred.idx <- as.numeric(names(pred))
        pred.idx <- as.numeric(rownames(X.conf))
        y.pred[pred.idx] <- pred
    }
    y.pred
}


# лучшая квантильная регрессия
# возвращаем обученную модель с квантилем tau.best
rq.tau.best <- function(X.conf, frmla='time~.', method='br'){ 
    taus <- seq(0.3,.7, by=0.01)
    fit.rq <- rq(as.formula(frmla), data=X.conf, tau=taus, method=method)
    mapes.tau <- sapply(data.frame(predict(fit.rq)), function(pred) mape(X.conf$time, pred))
    tau.best <- fit.rq$tau[which.min(mapes.tau)]
    fit.rq <- rq(as.formula(frmla), data=X.conf, tau=tau.best, method=method)
    
    fit.rq
}

# обучаем стек моделей
stack.train <- function(X.train){
    models = list()
    for (i in 1:length(confs)){
        # cat(i, '\n')
        X.conf <- subset(X.train, conf==confs[i])
        X.conf$conf <- NULL
        # считаем модель и метрику
        # выбираем лучшую модель по метрике
        # базовая линейная модель
        fit <- lm(time ~ ., data=X.conf)
        mape.best <- mape(X.conf$time, predict(fit))
        # модель rlm mn+nk+mnk
        fit.new <- rlm(time~m:n + n:k + m:n:k, data=X.conf)
        mape.new <- mape(X.conf$time, predict(fit.new))
        if (mape.new < mape.best){
            fit <- fit.new
            mape.best <- mape.new
        }
        # rlm mnk
        fit.new <- rlm(time~., data=X.conf)
        mape.new <- mape(X.conf$time, predict(fit.new))
        if (mape.new < mape.best){
            fit <- fit.new
            mape.best <- mape.new
        }
        # модель rq mn+nk+mnk best.tau
        frmla <- 'time~m:n + n:k + m:n:k'
        fit.new <- rq.tau.best(X.conf, frmla)
        mape.new <- mape(X.conf$time, predict(fit.new))
        if (mape.new < mape.best){
            fit <- fit.new
            mape.best <- mape.new
        }
        # модель rq .mn+nk best.tau
        frmla <- 'time~. + m:n + n:k'
        fit.new <- rq.tau.best(X.conf, frmla)
        mape.new <- mape(X.conf$time, predict(fit.new))
        if (mape.new < mape.best){
            fit <- fit.new
            mape.best <- mape.new
        }
        # rq mnk best.tau
        fit.new <- rq.tau.best(X.conf)
        mape.new <- mape(X.conf$time, predict(fit.new))
        if (mape.new < mape.best){
            fit <- fit.new
            mape.best <- mape.new
        }
        
        models[[confs[i]]] <- fit
    }
    
    models
}


# предсказываем y
get.y.pred <- function(X) {
    y <- rep(1, nrow(X))
    for (i in 1:length(confs)){
        X.conf <- subset(X, conf==confs[i])
        X.conf$conf <- NULL
        # y.pred <- predict(models[[confs[i]]], newdata = X.conf)
        if ( paste(class(models[[confs[i]]]), collapse=' ')=='xgb.Booster' ) {
            y.pred <- predict(models[[confs[i]]], newdata = as.matrix(X.conf))
        } else {
            y.pred <- predict(models[[confs[i]]], newdata = X.conf)
        }
        pred.idx <- as.numeric(rownames(X.conf))
        y[pred.idx] <- y.pred
    }
    y[as.numeric(rownames(X))]
}







# для тестирования возможных моделей
# обучаем стек моделей
stack.train.try <- function(X.train){
    models = list()
    for (i in 1:length(confs)){
        # cat(i, '\n')
        X.conf <- subset(X.train, conf==confs[i])
        X.conf$conf <- NULL
        # считаем модель и метрику
        # выбираем лучшую модель по метрике
        # базовая линейная модель
        fit <- lm(time ~ ., data=X.conf)
        mape.best <- mape(X.conf$time, predict(fit))
        # rlm mnk
#         fit.new <- rlm(time~., data=X.conf)
#         mape.new <- mape(X.conf$time, predict(fit.new))
#         if (mape.new < mape.best){
#             fit <- fit.new
#             mape.best <- mape.new
#         }
#         # rq mnk best.tau
#         fit.new <- rq.tau.best(X.conf)
#         mape.new <- mape(X.conf$time, predict(fit.new))
#         if (mape.new < mape.best){
#             fit <- fit.new
#             mape.best <- mape.new
#         }
        # rlm mnk (без m,n,k по отдельности)
        fit.new <- rlm(time~mkn, data=X.conf)
        mape.new <- mape(X.conf$time, predict(fit.new))
        if (mape.new < mape.best){
            fit <- fit.new
            mape.best <- mape.new
        }
        # модель rq mkn best.tau (без m,n,k по отдельности)
        frmla <- 'time ~ mkn'
        fit.new <- rq.tau.best(X.conf, frmla)
        mape.new <- mape(X.conf$time, predict(fit.new))
        if (mape.new < mape.best){
            fit <- fit.new
            mape.best <- mape.new
        }
#         # rq mnk best.tau + n/k + k/m
#         frmla <- 'time~. + n/k + k/m'
#         fit.new <- rq.tau.best(X.conf, frmla)
#         mape.new <- mape(X.conf$time, predict(fit.new))
#         if (mape.new < mape.best){
#             fit <- fit.new
#             mape.best <- mape.new
#         }
#         
#         # модель rlm mn+nk+mnk
#         fit.new <- rlm(time~m:n + n:k + m:n:k, data=X.conf)
#         mape.new <- mape(X.conf$time, predict(fit.new))
#         if (mape.new < mape.best){
#             fit <- fit.new
#             mape.best <- mape.new
#         }
#         # rlm mnk
#         fit.new <- rlm(time~., data=X.conf)
#         mape.new <- mape(X.conf$time, predict(fit.new))
#         if (mape.new < mape.best){
#             fit <- fit.new
#             mape.best <- mape.new
#         }

        
        models[[confs[i]]] <- fit
    }
    
    models
}

get.cutoff <- function(y, y.train){
    cutoffs <- seq(0.85, 1.4, by=0.001)
    df.cutoff <- data.frame(cutoff=cutoffs, mape.fixed=0)
    for (cutoff.low in cutoffs){
        y.train.fixed <- y.train
        y.train.fixed[y.train<cutoff.low] <- cutoff.low
        df.cutoff$mape.fixed[df.cutoff$cutoff==cutoff.low] <- mape(y, y.train.fixed)
    }
    
    df.cutoff$cutoff[which.min(df.cutoff$mape.fixed)]
}


MAPE <- function (data, lev = NULL, model = NULL) { 
    out <- c(defaultSummary(data, lev = NULL, model = NULL))
    MAPE <- sum(abs(data$obs-data$pred) / data$obs) / length(data$obs)
    c(out,MAPE=MAPE)
} 


# дайте лучший бустинг
get.xgb.conf <- function(X.conf, seed=19){
    set.seed(seed)
    cv.ctrl <- trainControl(method = "repeatedcv", repeats = 5, number = 5, 
                            allowParallel=TRUE, summaryFunction = MAPE)
    xgb.grid.linear <- expand.grid(nrounds = seq(3, 30, by=1),
                                   lambda=seq(0.5,2, by=0.1),
                                   alpha=seq(0,1, by=1)
    )
    xgb.tune <-train(time~.,
                     data=X.conf,
                     method="xgbLinear",
                     trControl=cv.ctrl,
                     tuneGrid=xgb.grid.linear,
                     verbose=FALSE,
                     metric="MAPE",
                     maximize=FALSE, 
                     nthread = 4
    )
    print(xgb.tune$finalModel$tuneValue)
    xgb.tune$finalModel
}





