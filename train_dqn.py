import cbpro 
from datetime import datetime, timedelta
import numpy as np 
import time
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import USD, BTC, ETH, LTC
import tensortrade.env.default as default
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
import tensorflow as tf 
from tensortrade.agents import DQNAgent, ParallelDQNAgent


public_client = cbpro.PublicClient()

now = datetime.now() 
delta = timedelta(days = 90)
start = now - delta
print(start)

def get_data_range(start, end, granularity, product):
    delta = timedelta(seconds=granularity)
    cur_time = start
    data = np.array([], dtype=np.float32).reshape(0,6)
    while cur_time < end:
        print(cur_time)
        cur_segment = public_client.get_product_historic_rates(product, start=cur_time, end=(cur_time + (delta * 300)), granularity=granularity)
        #print(len(cur_segment))
        cur_time = cur_time + (delta * len(cur_segment))
        #print(cur_time)
        cur_segment = np.flip(np.array(cur_segment), axis=0)
        #print(cur_segment.shape)
        data = np.concatenate((data, cur_segment), axis=0)
        print(data.shape)
        time.sleep(0.34)
    return data 


ETH_USD = get_data_range(start, now, 900, 'ETH-USD')
print('done')
BTC_USD = get_data_range(start, now, 900, 'BTC-USD')
print('done')
ETH_BTC = get_data_range(start, now, 900, 'ETH-BTC')
print('done')


def setup_env(ETH_USD, BTC_USD, ETH_BTC):
    coinbase = Exchange("Coinbase", service=execute_order)(
        Stream.source(ETH_USD[:, 4] , dtype="float").rename("USD-ETH"),
        Stream.source(BTC_USD[:, 4], dtype="float").rename("USD-BTC"),
        #Stream.source(ETH_BTC[:, 4], dtype="float").rename("ETH-BTC")
    )
    with NameSpace("coinbase"):
        coinbase_streams = [
            Stream.source(ETH_USD[:, 0] , dtype="float").rename("ETH:date"),
            Stream.source(ETH_USD[:, 1] , dtype="float").rename("ETH:open"),
            Stream.source(ETH_USD[:, 2] , dtype="float").rename("ETH:high"),
            Stream.source(ETH_USD[:, 3] , dtype="float").rename("ETH:low"),
            Stream.source(ETH_USD[:, 4] , dtype="float").rename("ETH:close"),
            Stream.source(ETH_USD[:, 5] , dtype="float").rename("ETH:volume"),
        
            Stream.source(BTC_USD[:, 0] , dtype="float").rename("BTC:date"),
            Stream.source(BTC_USD[:, 1] , dtype="float").rename("BTC:open"),
            Stream.source(BTC_USD[:, 2] , dtype="float").rename("BTC:high"),
            Stream.source(BTC_USD[:, 3] , dtype="float").rename("BTC:low"),
            Stream.source(BTC_USD[:, 4] , dtype="float").rename("BTC:close"),
            Stream.source(BTC_USD[:, 5] , dtype="float").rename("BTC:volume"),
        
            #Stream.source(ETH_BTC[:, 0] , dtype="float").rename("ETH_BTC:date"),
            #Stream.source(ETH_BTC[:, 1] , dtype="float").rename("ETH_BTC:open"),
            #Stream.source(ETH_BTC[:, 2] , dtype="float").rename("ETH_BTC:high"),
            #Stream.source(ETH_BTC[:, 3] , dtype="float").rename("ETH_BTC:low"),
            #Stream.source(ETH_BTC[:, 4] , dtype="float").rename("ETH_BTC:close"),
            #Stream.source(ETH_BTC[:, 5] , dtype="float").rename("ETH_BTC:volume"),

            #Stream.source(BTC_USD[:, 1:], dtype="float").rename("BTC-USD"),
            #Stream.source(ETH_BTC[:, 1:], dtype="float").rename("ETH-BTC")
        ]
    feed = DataFeed(coinbase_streams)

    portfolio = Portfolio(USD, [
        Wallet(coinbase, 3000 * USD),
        Wallet(coinbase, 0.01 * BTC),
        Wallet(coinbase, 0.3 * ETH),
    ])

    renderer_feed = DataFeed([
        Stream.source(ETH_USD[:, 0] , dtype="float").rename("date"),
        Stream.source(ETH_USD[:, 1] , dtype="float").rename("open"),
        Stream.source(ETH_USD[:, 2] , dtype="float").rename("high"),
        Stream.source(ETH_USD[:, 3] , dtype="float").rename("low"),
        Stream.source(ETH_USD[:, 4] , dtype="float").rename("close"),
        Stream.source(ETH_USD[:, 5] , dtype="float").rename("volume"),
    ])

    env = default.create(
        portfolio=portfolio,
        action_scheme="managed-risk",
        reward_scheme="risk-adjusted",
        feed=feed,
        renderer_feed=renderer_feed,
        renderer=default.renderers.PlotlyTradingChart(),
        window_size=20
    )

    return env


test_steps  = 1000
env = setup_env(ETH_USD[0:-test_steps, :], BTC_USD[0:-test_steps, :], ETH_BTC[0:-test_steps, :]) 

policy_network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=env.observation_space.shape),
            tf.keras.layers.Flatten(), 
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(512),
            tf.keras.layers.Activation('swish'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
    
            tf.keras.layers.Dense(256),
            tf.keras.layers.Activation('swish'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
    
            tf.keras.layers.Dense(128),
            tf.keras.layers.Activation('swish'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),

            #tf.keras.layers.Dense(64),
            #tf.keras.layers.Activation('swish'),
            #tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(env.action_space.n)
        ])

policy_network.summary()

agent = DQNAgent(env, policy_network)
print(agent)
mean_reward= agent.train(n_steps=8000, n_episodes=30, save_path="agents/", update_target_every=10, memory_capacity=10000, eps_decay_steps=2000, render_interval=4000, discount_factor=0.991)
print(mean_reward)



def evaluate(agent, test_env : 'TradingEnv'): 
    done = False
    state = test_env.reset()
    total_reward = 0
    total_steps = 0
    while not done: 
        action = agent.get_action(state)
        state, reward, done, _ = test_env.step(action)
        total_reward += reward
        total_steps += 1

    return total_steps, total_reward, total_reward / total_steps 

test_env = setup_env(ETH_USD[-test_steps:, :], BTC_USD[-test_steps:, :], ETH_BTC[0:-test_steps, :]) 
total_test_steps, test_reward, test_mean_reward = evaluate(agent, test_env)

print(total_test_steps)
print(test_reward)
print(test_mean_reward)


