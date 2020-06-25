# gym_minigrid

Centralized repo containing all Minigrid environments used and created for the ASIST project

## Getting Started

This repo is intended to be used as a submodule in other repositories. To add this repo as a submodule:
```
git submodule add https://gitlab.com/cmu_asist/gym_minigrid
```

## In Other Projects

### Cloning

To clone another project that uses this repo as a submodule, just pass `--recurse-submodules`:
```
git clone --recurse-submodules [main_repo_url]
```

If you've already cloned the repo but forgot to pass `--recurse-submodules`, the directory `gym_minigrid` will be empty.
To initialize it, navigate to `gym_minigrid` and do the following:
```
git submodule update --init
```

### Pushing & Pulling
By default, changes to this repo as a submodule need to be handled separately from the parent module. 
To push/pull the submodule, navigate to `gym_minigrid` and `git push` or `git pull`.

To pull changes from both the main repo and the submodule repo simultaneously, pass `--recurse-submodules` from the main project root:
```
git pull --recurse-submodules
```
