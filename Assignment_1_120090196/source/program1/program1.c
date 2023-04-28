#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

char *signal_table[] = {
	"", // 0
	"SIGHUP", // 1
	"SIGINT", // 2
	"SIGQUIT", // 3
	"SIGILL", // 4
	"SIGTRAP", // 5
	"SIGABRT", // 6
	"SIGBUS", // 7
	"SIGFPE", // 8
	"SIGKILL", // 9
	"SIGUSR1", // 10
	"SIGSEGV", // 11
	"SIGUSR2", // 12
	"SIGPIPE", // 13
	"SIGALRM", // 14
	"SIGTERM", // 15
};

int main(int argc, char *argv[])
{
	if (argc < 2) {
		perror("please input correct arguments");
		exit(1);
	}

	/* fork a child process */
	printf("Process start to fork\n");
	pid_t pid = fork();
	if (pid == -1) {
		perror("fork");
		exit(1);
	}

	if (pid == 0) {
		// child
		printf("I'm the Child Process, my pid = %d\n", getpid());
		printf("Child process start to execute test program:\n");
		/* execute test program */
		execve(argv[1], argv + 2, NULL);
	} else {
		// parent
		printf("I'm the Parent Process, my pid = %d\n", getpid());
		/* wait for child process terminates */
		int sts;
		waitpid(pid, &sts, WUNTRACED);

		printf("Parent process receives SIGCHLD signal\n");

		/* check child process'  termination status */
		if (WIFEXITED(sts)) {
			printf("Normal termination with EXIT STATUS = %d\n",
			       WEXITSTATUS(sts));
		} else if (WIFSIGNALED(sts)) {
			int signal = WTERMSIG(sts);
			printf("child process get %s signal\n",
			       signal_table[signal]);
		} else if (WIFSTOPPED(sts)) {
			printf("child process get SIGSTOP signal\n");
		} else {
			printf("CHILD PROCESS CONTINUED\n");
		}
	}
}
