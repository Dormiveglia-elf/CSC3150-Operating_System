#include <linux/err.h>
#include <linux/fs.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kmod.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/pid.h>
#include <linux/printk.h>
#include <linux/sched.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");

#define WTERMSIG(status) ((status)&0x7f)
#define WIFEXITED(status) (WTERMSIG(status) == 0)
#define WIFSIGNALED(status) (((signed char)(((status)&0x7f) + 1) >> 1) > 0)
#define WIFSTOPPED(status) (((status)&0xff) == 0x7f)

struct wait_opts {
	enum pid_type wo_type;
	int wo_flags;
	struct pid *wo_pid;

	struct waitid_info *wo_info;
	int wo_stat;
	struct rusage *wo_rusage;

	wait_queue_entry_t child_wait;
	int notask_error;
};

extern long do_wait(struct wait_opts *wo);
extern struct filename *getname_kernel(const char *filename);
extern pid_t kernel_clone(struct kernel_clone_args *kargs);
extern int do_execve(struct filename *filename,
		     const char __user *const __user *__argv,
		     const char __user *const __user *__envp);
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

char *error_msg[] = {
	"", // 0
	"hung up", // 1
	"terminal interrupt", // 2
	"terminal quit", // 3
	"illegal", // 4
	"trap error", // 5
	"abort error", // 6
	"bus error", // 7
	"float error", // 8
	"is killed", // 9
	"User Signal 1", // 10
	"segmentation fault error", // 11
	"User Signal 2", // 12
	"pipe error", // 13
	"alarm", // 14
	"terminated", // 15
};

int my_exec(void)
{
	int result;
	// path should be /tmp/test before submitting
	const char path[] = "/tmp/test";

	// get filename
	struct filename *my_filename = getname_kernel(path);
	printk("[program2] : child process\n");
	// execute the test program
	result = do_execve(my_filename, NULL, NULL);
	if (!result) {
		return 0;
	}

	// if exec failed
	do_exit(result);
}

void my_wait(pid_t pid)
{
	int status;
	int signal_id = 0;
	// look up a PID from hash table and retyrn with it's count evaluated
	struct pid *wo_pid = find_get_pid(pid);
	struct wait_opts wo = {
		.wo_type = PIDTYPE_PID,
		.wo_pid =
			wo_pid, // Kernel's internal notion of a process identifier. 
		.wo_flags =
			WEXITED |
			WUNTRACED, // Wait options. (0, WNOHANG, WEXITED, etc.)
		.wo_info = NULL, // Singal information.
		.wo_rusage = NULL, // Resource usage
	};

	do_wait(&wo);

	// catch the child's signal and print out message
	status = wo.wo_stat;
	signal_id = WTERMSIG(status);
	if (WIFEXITED(status)) {
		printk("[program2] : child process exit normally\n");
	} else if (WIFSIGNALED(status)) {
		printk("[program2] : get %s signal\n", signal_table[signal_id]);
		printk("[program2] : child process %s\n", error_msg[signal_id]);
	} else if (WIFSTOPPED(status)) {
		// stopped
		signal_id = ((status & 0xff00) >> 8);
		printk("[program2] : get STOPPED signal\n");
		printk("[program2] : child process stopped\n");
	} else {
		// continue
		printk("[program2] : get SIGCONT signal\n");
		printk("[program2] : child process continued\n");
	}

	printk("[program2] : The return signal is %d\n", signal_id);
}

// implement fork function
int my_fork(void *argc)
{
	// set default sigaction for current process
	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	pid_t pid;
	struct kernel_clone_args kernel_clone_args_ = {
		.flags = SIGCHLD,
		.exit_signal = SIGCHLD,
		.stack = (unsigned long)&my_exec,
		.stack_size = 0,
		.parent_tid = NULL,
		.child_tid = NULL,
		.tls = 0,
	};

	for (i = 0; i < _NSIG; i++) {
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}

	/* fork a process using kernel_clone or kernel_thread */
	pid = kernel_clone(&kernel_clone_args_);

	/* execute a test program in child process */
	// Print out the process id for both parent and child process.
	printk("[program2] : The child process has pid = %d\n", pid);
	printk("[program2] : This is the parent process, pid = %d\n",
	       current->pid);

	/* wait until child process terminates */
	my_wait(pid);

	return 0;
}

static int __init program2_init(void)
{
	struct task_struct *task;
	printk("[program2] : module_init {Zhenyu PAN} {120090196}\n");

	/* write your code here */

	/* create a kernel thread to run my_fork */
	printk("[program2] : module_init create kthread start\n");
	task = kthread_create(&my_fork, NULL, "kthread");
	// wake up new thread if ok
	if (!IS_ERR(task)) {
		// start to run my_fork function
		printk("[program2] : module_init kthread start\n");
		wake_up_process(task);
	}
	return 0;
}

static void __exit program2_exit(void)
{
	printk("[program2] : module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);
