library(dplyr)
library(ggplot2)

# Load data
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
