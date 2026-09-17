# Git

Read [this](https://rogerdudler.github.io/git-guide/) guide

I'll type the commands that goes with each step, but it is good to understand the basics. Read it till `update & merge`, but do the first entry of `git pull`. For desktop gui clients, text me (luuk).

## Branches

We work on an unique branch for every task. This is to make sure that our work doesn't interfere with each other. If you create a new branch for a task, make sure you create it from the main branch.

```bash
git switch master
git pull    # if that doesn't work $ git pull origin master
git switch -c <your-branch-name>   # -c stands for create if that helps with remembering
```

The branch names should be lowercase and separated with hyphen's (-). So like you see in the example `your-branch-name`

## Commits

Each commit should be descriptive of what you did. This makes it easier for us to search for changes we made that we might wanna revert or whatever. Doesn't have to be anything crazy. Since we work with defined tasks for each branch, one commit will likely be fine for each feature we work on.

```bash
git add --all    # adds all changes (tracked or untracked) to the staging area
git commit -m "your git commit message"
git push origin <your-branch-name>
```

After that send me (luuk) a text message that i can review it and i can take it from there
