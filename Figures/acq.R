set.seed(0)

ls <- 1
N <- 500
x <- matrix(seq(-5, 5, length = N), N, 1)

dist2 <- as.matrix(dist(x))^2
K <- 2 * exp(-0.5 * dist2 * ls) + diag(N) * 1e-6
L <- chol(K)

x_train <- matrix(c(-4, -1, 3), 3, 1)
y_train <- matrix(c(0.5, -.4, -.7), 3, 1)

x_train_i <- x_train
y_train_i <- y_train

i <- 3
dist2 <- as.matrix(dist(x_train_i))^2
K <- 2 * exp(-0.5 * dist2 * ls) + diag(i) * 1e-6
dist2 <- matrix(x^2, N, i) - 2 * x %*% t(x_train_i) + matrix(x_train_i^2, N, i, byrow = TRUE)
Kcross <- 2 * exp(-0.5 * dist2 * ls) 
dist2 <- as.matrix(dist(x))^2
Ktest <- 2 * exp(-0.5 * dist2 * ls) + diag(N) * 1e-6
Sigma <- Ktest - Kcross %*% chol2inv(chol(K)) %*% t(Kcross)

L <- chol(Sigma)
m <- Kcross %*% chol2inv(chol(K)) %*% y_train_i
std <- sqrt(diag(Sigma)) 

colors = c("red", "blue", "green3")

pdf(paste("gp.pdf", sep = ""), width = 7, height = 4)
	par(mar = c(2.0,2.0,0,0))
	plot(x, m, type = "l", lwd = 7, col = "black", ylim = c(-1.5, 1.75), xlab = "x", ylab = "f(x)")

	for (j in 1 : (length(x) - 1)) {
		polygon(c(x[ j ], x[ j ], x[ j + 1 ], x[ j  + 1 ]), 
			c(m[ j ] - std[ j ], m[ j ] + std[ j ], m[ j ] + std[ j ], m[ j ] - std[ j ]), col = rgb(0.55, 0.7, 1.0), border = NA)	
	}

	lines(x, m, col = "black", lwd = 7)
	lines(x, m + std, col = "black", lwd = 1, lty = 1)
	lines(x, m - std, col = "black", lwd = 1, lty = 1)
	points(x_train_i, y_train_i, pch = 1, lwd = 5, col = "red", cex = 2)

dev.off()

v <- min(y_train)

pdf(paste("pi.pdf", sep = ""), width = 7, height = 1)
	par(mar = c(0.0,2.0,0,0))
	values <- pnorm((v - m) / std)
	plot(x, values, ylim = c(0, max(values)), col = "black", lwd = 3, type = "line")
	for (j in 1 : (length(x) - 1)) {
		polygon(c(x[ j ], x[ j ], x[ j + 1 ], x[ j  + 1 ]), 
			c(0, values[ j ], values[ j ], 0), col = "blue", border = NA)
	}
	lines(x, values, ylim = c(0, max(values)), col = "black", lwd = 5, type = "line")
dev.off()


pdf(paste("ei.pdf", sep = ""), width = 7, height = 1)
	par(mar = c(0.0,2.0,0,0))
	gamma <- ((v - m) / std)
	values <- std * (gamma * pnorm(gamma) + dnorm(gamma))
	
	plot(x, values, ylim = c(0, max(values)), col = "black", lwd = 3, type = "line")
	for (j in 1 : (length(x) - 1)) {
		polygon(c(x[ j ], x[ j ], x[ j + 1 ], x[ j  + 1 ]), 
			c(0, values[ j ], values[ j ], 0), col = "blue", border = NA)
	}
	lines(x, values, ylim = c(0, max(values)), col = "black", lwd = 5, type = "line")
dev.off()


pdf(paste("ucb.pdf", sep = ""), width = 7, height = 1)
	par(mar = c(0.0,2.0,0,0))
	gamma <- ((v - m) / std)
	values <- -(m - std) - min(-(m - std))
	
	plot(x, values, ylim = c(0, max(values)), col = "black", lwd = 3, type = "line")
	for (j in 1 : (length(x) - 1)) {
		polygon(c(x[ j ], x[ j ], x[ j + 1 ], x[ j  + 1 ]), 
			c(0, values[ j ], values[ j ], 0), col = "blue", border = NA)
	}
	lines(x, values, ylim = c(0, max(values)), col = "black", lwd = 5, type = "line")
dev.off()


### We find the values 

x <- x[ seq(1, N, 5) ]
means_pred <- m[ seq(1, N, 5) ]
std_pred <- std[ seq(1, N, 5) ]
N_samples <- 100
N <- N / 5
values <- rep(0, N)
samples <- rnorm(N_samples)
samples_min <- matrix(rnorm(1e3 * N), N, 1e3)

dist2 <- as.matrix(dist(x_train))^2
K <- 2 * exp(-0.5 * dist2 * ls) + diag(3) * 1e-6
dist2 <- matrix(x^2, N, 3) - 2 * x %*% t(x_train) + matrix(x_train^2, N, 3, byrow = TRUE)
Kcross <- 2 * exp(-0.5 * dist2 * ls) 
dist2 <- as.matrix(dist(x))^2
Ktest <- 2 * exp(-0.5 * dist2 * ls) + diag(N) * 1e-6
Sigma <- Sigmao <- Ktest - Kcross %*% chol2inv(chol(K)) %*% t(Kcross)
L <- Lo <- chol(Sigma)
m <- mo <- Kcross %*% chol2inv(chol(K)) %*% y_train

sample_generated <- sample_generated_o <- t(L) %*% samples_min + matrix(m, N, 1e3)
counts <- table(apply(sample_generated, 2, which.min))
obs <- rep(0, N)
obs[ as.integer(rownames(counts)) ] <- counts
obs <- obs / sum(obs)
Ent0 <- -obs * log(obs)
Ent0[ is.nan(Ent0) ] <- 0
Ent0 <- sum(Ent0)

for (i in 1 : N) {

	value_tmp <- 0
#	samples <- rnorm(N_samples)

	for (s in 1 : N_samples) {

#		samples_min <- matrix(rnorm(1e3 * N), N, 1e3)

		y_new <- samples[ s ] * std_pred[ i ] + means_pred[ i ]
		x_new <- x[ i ]

		x_train_i <- rbind(x_train, matrix(x_new, 1, 1))
		y_train_i <- rbind(y_train, matrix(y_new, 1, 1))

		dist2 <- as.matrix(dist(x_train_i))^2
		K <- 2 * exp(-0.5 * dist2 * ls) + diag(4) * 1e-6
		dist2 <- matrix(x^2, N, 4) - 2 * x %*% t(x_train_i) + matrix(x_train_i^2, N, 4, byrow = TRUE)
		Kcross <- 2 * exp(-0.5 * dist2 * ls) 
		dist2 <- as.matrix(dist(x))^2
		Ktest <- 2 * exp(-0.5 * dist2 * ls) + diag(N) * 1e-6
		Sigma <- Ktest - Kcross %*% chol2inv(chol(K)) %*% t(Kcross)

		L <- chol(Sigma)
		m <- Kcross %*% chol2inv(chol(K)) %*% y_train_i

		sample_generated <- t(L) %*% samples_min + matrix(m, N, 1e3)
		counts <- table(apply(sample_generated, 2, which.min))
		obs <- rep(0, N)
		obs[ as.integer(rownames(counts)) ] <- counts
		obs <- obs / sum(obs)

		Ent <- -obs * log(obs)
		Ent[ is.nan(Ent) ] <- 0
		Ent <- sum(Ent)

		value_tmp <- value_tmp + Ent
	}

	avg_ent <- value_tmp / N_samples

	values[ i ] <- Ent0 - avg_ent

	print(i)
}


pdf(paste("es.pdf", sep = ""), width = 7, height = 1)
	par(mar = c(0.0,2.0,0,0))
	par(lwd = 3)
	barplot(values, col = "blue", space = 0, border = "black")
dev.off()

