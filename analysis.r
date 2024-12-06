library(dplyr)
library(ggplot2)


## Grid search analysis
df <- read.csv('metrics/grid_rewards.csv')
colnames(df)


# Plots
mr <- df %>% 
    group_by(tau) %>%
    summarise(rew=mean(avg_reward))
ggplot(data=df) +
    geom_jitter(aes(x=tau, y=avg_reward), width=1) +
    geom_segment(aes(x=mr$tau[1], y=mr$rew[1], xend=mr$tau[2], yend=mr$rew[2]), col='blue') +
    geom_segment(aes(x=mr$tau[2], y=mr$rew[2], xend=mr$tau[3], yend=mr$rew[3]), col='blue') +
    geom_segment(aes(x=mr$tau[3], y=mr$rew[3], xend=mr$tau[4], yend=mr$rew[4]), col='blue') +
    geom_segment(aes(x=mr$tau[4], y=mr$rew[4], xend=mr$tau[5], yend=mr$rew[5]), col='blue') +
    labs(title='Average Reward for each Tau', x='Tau', y='Reward')
ggsave('figures/tau_grid_search.png', width=7, height=5)

mr <- df %>% 
    group_by(kappa) %>%
    summarise(rew=mean(avg_reward))
ggplot(data=df) +
    geom_jitter(aes(x=kappa, y=avg_reward), width=0.1) +
    geom_segment(aes(x=mr$kappa[1], y=mr$rew[1], xend=mr$kappa[2], yend=mr$rew[2]), col='blue') +
    geom_segment(aes(x=mr$kappa[2], y=mr$rew[2], xend=mr$kappa[3], yend=mr$rew[3]), col='blue') +
    geom_segment(aes(x=mr$kappa[3], y=mr$rew[3], xend=mr$kappa[4], yend=mr$rew[4]), col='blue') +
    geom_segment(aes(x=mr$kappa[4], y=mr$rew[4], xend=mr$kappa[5], yend=mr$rew[5]), col='blue') +
    labs(title='Average Reward for each Kappa', x='Kappa', y='Reward') +
    scale_x_log10()
ggsave('figures/kappa_grid_search.png', width=7, height=5)


mr <- df %>% 
    group_by(beta) %>%
    summarise(rew=mean(avg_reward))
ggplot(data=df) +
    geom_jitter(aes(x=beta, y=avg_reward), width=0.0002) +
    geom_segment(aes(x=mr$beta[1], y=mr$rew[1], xend=mr$beta[2], yend=mr$rew[2]), col='blue') +
    geom_segment(aes(x=mr$beta[2], y=mr$rew[2], xend=mr$beta[3], yend=mr$rew[3]), col='blue') +
    geom_segment(aes(x=mr$beta[3], y=mr$rew[3], xend=mr$beta[4], yend=mr$rew[4]), col='blue') +
    geom_segment(aes(x=mr$beta[4], y=mr$rew[4], xend=mr$beta[5], yend=mr$rew[5]), col='blue') +
    labs(title='Average Reward for each Beta', x='Beta', y='Reward')
ggsave('figures/beta_grid_search.png', width=7, height=5)


mr <- df %>% 
    group_by(sigma) %>%
    summarise(rew=mean(avg_reward))
ggplot(data=df) +
    geom_jitter(aes(x=sigma, y=avg_reward), width=0.2) +
    geom_segment(aes(x=mr$sigma[1], y=mr$rew[1], xend=mr$sigma[2], yend=mr$rew[2]), col='blue') +
    geom_segment(aes(x=mr$sigma[2], y=mr$rew[2], xend=mr$sigma[3], yend=mr$rew[3]), col='blue') +
    geom_segment(aes(x=mr$sigma[3], y=mr$rew[3], xend=mr$sigma[4], yend=mr$rew[4]), col='blue') +
    geom_segment(aes(x=mr$sigma[4], y=mr$rew[4], xend=mr$sigma[5], yend=mr$rew[5]), col='blue') +
    labs(title='Average Reward for each Sigma', x='Sigma', y='Reward')
ggsave('figures/sigma_grid_search.png', width=7, height=5)


mr <- df %>% 
    group_by(delta_t) %>%
    summarise(rew=mean(avg_reward))
ggplot(data=df) +
    geom_jitter(aes(x=delta_t, y=avg_reward), width=0.1) +
    geom_segment(aes(x=mr$delta_t[1], y=mr$rew[1], xend=mr$delta_t[2], yend=mr$rew[2]), col='blue') +
    geom_segment(aes(x=mr$delta_t[2], y=mr$rew[2], xend=mr$delta_t[3], yend=mr$rew[3]), col='blue') +
    geom_segment(aes(x=mr$delta_t[3], y=mr$rew[3], xend=mr$delta_t[4], yend=mr$rew[4]), col='blue') +
    geom_segment(aes(x=mr$delta_t[4], y=mr$rew[4], xend=mr$delta_t[5], yend=mr$rew[5]), col='blue') +
    labs(title='Average Reward for each Delta t', x='Delta t', y='Reward')
ggsave('figures/delta_t_grid_search.png', width=7, height=5)

