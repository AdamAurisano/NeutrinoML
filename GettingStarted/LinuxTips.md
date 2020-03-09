# Linux Tips

## Using Tab as auto-completion. 
If you know the name of the folder you are looking for in your directory, you cans simply click the first letter of that folder and hit tab. Linux terminal will then suggest all the possible options starting with the string you typed.

## Going backwards.
You can go back to the last directory by doing the following:
```
cd -
```

## Going home.
You can go back to the home directory by doing the following:
```
cd ~
```
or simply
```
cd
```

## Where are you?
To find out the address to your current destination, do the following:
```
pwd
```

## What's in here.
To find out what the contents are of the current directory, do the following:
```
ls
```
which is also equivalent to
```
ls -l
```

or to find out the contents of the top director, do the following:
```
ls /
```

# Multitasking
You can run multiple commands in one line of command
```
command_1; commmand_2; command_3
```
Let's say you want to run the next command only if the first one was successful, use && separator
```
command_1 && command_2
```

# I forgot
If you forgot the long line of command and can't remember the exact code, you can reverse search for it
```
ctrl+r
```
and type the part of the command you remember. It will automatically search your history for you.

# Frozen screen
If you ended with a frozen terminal, do `Ctrl+Q` to unfreeze it.

# Reading a log file
To read a log file and to track its process, do the following:
```
tail -f (path_to_Log)
```
To find the path to log, refer to `Where are you?`.

# Reuse
Let's create a directory. Now go into the newly created directory. You can use the argument of the previous command to do so.
```
ls (name of directory)
cd !$
```
You are now in the directory you have previously used as an argument.

# Stop it.
If you ran a command but you want to exit, you can press `Ctrl+C`.

# Help me.
Almost all command come with a help page. If you'd like to see that page do the following:
```
(command_tool) -- help
```
