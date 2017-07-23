# PPO_pendulmn

Swinging up a [pendulmn](https://gym.openai.com/envs/Pendulum-v0) by [PPO(Proximal Policy Optimization)](https://blog.openai.com/openai-baselines-ppo/)

![movie](https://raw.githubusercontent.com/ashitani/PPO_pendulmn/master/movie/out.gif)

# Usage

## train

```
python run_pendulumn.py train
```

## replay

```
python run_pendulumn.py replay
```

If you need movie (animation gif) file, set RECORD\_MOVIE=True in the script run\_pendulmn.py to output serial numbered png file and exec:

```
cd movie
./conv.sh
```

## plot learning curve

```
python plot_log.py
```

# License

MIT
