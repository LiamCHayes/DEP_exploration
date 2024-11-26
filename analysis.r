library(dplyr)
library(ggplot2)


# Grid search analysis
df <- read.csv('metrics/grid_rewards.csv')

# Plots
ggplot_data <- df %>% 
    filter(tau == 13)
ggplot(data=ggplot_data) +
    geom_point(aes(x=X, y=avg_reward)) + 
    geom_vline(xintercept = 1035)

# Statistics
high_reward <- df %>% 
    arrange(desc(avg_reward))

high_reward[seq(10),] %>% filter(tau == 13)

# Best parameters
# tau = 13
# kappa = 1000
# beta = 0.0025
# sigma = 5.25
# delta_t <= 2


# Deep model matrix analysis, ten thousand episode run
df <- read.csv('dep_deep_backprop_results/best_params/metrics.csv')

# Only take rewards bigger than 50
df_high_reward <- df %>%
    filter(reward > 20)

ggplot(data=df) +
    geom_point(aes(x=X, y=reward)) 

ggplot(data=df_high_reward) +
    geom_point(aes(x=X, y=reward))

plot(df_high_reward$X)

# First layer dep init
df <- read.csv('dep_layer_results/init/metrics.csv')
df
ggplot(data=df) +
    geom_point(aes(x=X, y=reward))
