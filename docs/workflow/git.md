# Git

Read [this](https://rogerdudler.github.io/git-guide/) guide

I'll type the commands that goes with each step, but it is good to understand the basics. Read it till (and not including) the `update & merge` section, but do read the first entry about `git pull` in this `update & merge` section. For desktop gui clients, chatgpt (or text me but ill probably also have to look it up).

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

When adding files to the staged area, please do make sure that you don't accidentally add a change you didn't want to make, like a logs that aren't in the `.gitignore`. To check what is in your staged area, use `git status`.

```bash
git add /path/to/your/code  # this works with folders too. If you have multiple files changed in src/models, `git add src/models` will add all of them into you staging area.
git commit -m "your git commit message"
git push origin <your-branch-name>
```

After that please make a pull request that requests to merge your branch into master. I'll review it, possibly ask you to change things and merge it into master.
