# Introduction to Git

This repository is managed using Git, which is a version control system. The purpose of this system is to maintain a shared base, and to properly version and maintain our code. After you initially clone this repository to your working area, changes you make to code locally will not automatically be propagated back up to the remote repository, nor will changes made by others be automatically propagated down to your local repository. In order to keep your local copy up to date, it is necessary to *push* your local changes, and *pull* down changes from others, which can be done using git commands.

It's possible that Git might ask you to identify yourself when you try to run a command, with an output that begins

`*** Please tell me who you are.`

You can make these changes by running

`git config --global user.email "your email address"`

`git config --global user.name "your name"`

## Useful commands

There are a few git commands that provide information on your local repository without changing anything:

`git status` will list all files which have been changed locally compared to the remote repository. Any files you have added, deleted or changed will be listed here.

`git diff` will list all changes to all modified files. Optionally, you can run `git diff <file>` to list only the changes to a specific file.

## Pulling and pushing

`git pull origin master` will pull down all recent commits from the remote repository and apply them to your local repository. If you have uncommitted changes that conflict with incoming commits, the pull command will fail. In general you should commit any changes as often as possible to avoid getting into this state, but if you do find yourself in this situation, there are ways around it.

`git stash` will temporarily "hide" your local changes and restore your working area to a clean state. This will allow you to run a pull command without issue. If you wish, you can permanently discard your local changes by running `git stash clear`. If you want to reapply your local changes, you can run `git stash pop` after pulling to reapply your local changes. Be warned that if your local changes are in conflict with changes pulled down from the repository, some surgery might be required to manually merge in your local changes.

`git add` will stage local changes before committing. If you run `git status` after making local changes, you will see that all your changes show up in red. The first stage in committing those changes back up to the repository is to add them with `git add`. Running `git add` without any arguments will add **all** modified files simulaneously. If you only want to commit changes from specific files, you can cherry-pick those files by doing `git add <file>`. Repeating your `git status` command after adding a file, you should see its status changes from red to green.

`git commit -m "commit message"` will **commit** all added changes. This commit will stamp all your changes as a specific "commit", or change, to the repository, and will set that commit as the "new normal" for the repository. When running `git commit`, you'll see any changes you previously added, which showed up in green with a `git status`, are now gone. That is because git no longer considers them "changes", and instead considers them part of the nominal state of the repository. The "commit message" part of the command should be a sentence or a handful of sentences briefly summarising what was changed in the commit – this comes in very useful for understanding a file's history in version control. We're all guilty of writing cryptic commit messages from time to time, but try to make this message both clear and concise. I promise your future self will appreciate it.

`git push origin master` is the last stage in pushing your changes to the remote repository. By this point you will have **added** your local changes, and then **committed** them to your local repository. At this point, your local commits are just that -- local. In order for them to be propagated up so others can pull them down, they need to be pushed to the repository. This can be done with a simple `git push origin master` command. Note that you should make sure to pull down remote changes before committing and pushing your own changes – if you try to `push` to a remote repository which has new commits, your push will be refused, and you will have to manually merge the local and remote commits, and then push the resulting merge to the repository.
