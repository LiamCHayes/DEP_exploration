library(dplyr)
library(ggplot2)
library(zoo)

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

## DEP RL analysis
df <- read.csv('dep_RL_results/noise_0.05/metrics.csv')
colnames(df)

ggplot(data=df) +
    geom_point(aes(x=X, y=reward)) + 
    labs(title='Reward During Training', x="Episodes", y='Reward')
ggsave('figures/DEPRL_small_reward.png', width=7, height=5)


## Updating model matrix results
df <- read.csv('dep_backprop_results/forthepaper_longer/metrics.csv')
colnames(df)

ggplot(data=df) +
    geom_point(aes(x=X, y=loss)) +
    labs(title="Backpropegation on the Model Matrix - Loss", x="Episode", y="Loss")
ggsave('figures/backprop_m_loss.png', width=7, height=5)

ggplot(data=df) +
    geom_point(aes(x=X, y=reward)) +
    labs(title="Backpropegation on the Model Matrix - Reward", x="Episode", y="Reward")
ggsave('figures/backprop_m_reward.png', width=7, height=5)
    
## Deep updating model matrix
df <- read.csv('dep_deep_backprop_results/forthepaper_longer/metrics.csv')
colnames(df)

ggplot(data=df) +
    geom_point(aes(x=X, y=loss)) +
    labs(title="Learning a Deep Model Matrix - Loss", x="Episode", y="Loss")
ggsave('figures/deep_backprop_m_loss_500.png', width=7, height=5)

ggplot(data=df) +
    geom_point(aes(x=X, y=reward)) +
    labs(title="Learning a Deep Model Matrix - Reward", x="Episode", y="Reward")
ggsave('figures/deep_backprop_m_reward_500.png', width=7, height=5)

# Moving averages
k <- 15
df <- df %>% 
    mutate(loss_ma = rollmean(loss, k=k, fill=T, align='right')) %>%
    mutate(reward_ma = rollmean(reward, k=k, fill=T, align='right')) %>%
    filter(reward_ma != 1)

ggplot(data=df) +
    geom_line(aes(x=X, y=loss_ma)) +
    labs(title="Deep Model Matrix - Loss Moving Average", x="Episode", y="Loss")
ggsave('figures/deep_m_loss_ma.png', width=7, height=5)

ggplot(data=df) +
    geom_line(aes(x=X, y=reward_ma)) +
    labs(title="Deep Model Matrix - Reward Moving Average", x="Episode", y="Reward")
ggsave('figures/deep_m_reward_ma.png', width=7, height=5)
