#include <assert.h>
#include <ctype.h>
#include <dirent.h>
#include <locale.h>
#include <memory.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <wchar.h>

#define false 0
#define true 1
#define MAX_NAME_LEN 128

// implement: -A -p -n -V -U
struct option {
	const char *name;
	const char *full_name;
	int *enabled;
};

int enable_ascii = false;
int enable_show_pids = false;
int enable_numeric_sort = false;
int enable_version = false;
int enable_utf8 = false;
const struct option options[] = { { "-A", "--ascii", &enable_ascii },
				  { "-p", "--show-pids", &enable_show_pids },
				  { "-n", "--numeric-sort",
				    &enable_numeric_sort },
				  { "-V", "--version", &enable_version },
				  { "-U", "--unicode", &enable_utf8 } };

const int kOptionNum = (int)sizeof(options) / sizeof(struct option);

struct process {
	pid_t pid;
	pid_t ppid;
	char name[MAX_NAME_LEN];
	char state;
	struct process *parent;
	struct process *child;
	struct process *next;
} root = { 1, 0, "systemd", 'X', NULL, NULL };

int get_opt(int argc, char *argv[])
{
	bool is_match = false;
	// travese all arguments
	for (int i = 1; i < argc; i++) {
		is_match = false;
		for (int j = 0; j < kOptionNum; j++) {
			if (strcmp(options[j].name, argv[i]) == 0 ||
			    strcmp(options[j].full_name, argv[i]) == 0) {
				*(options[j].enabled) = true;
				is_match = true;
				break;
			}
		}
		if (!is_match)
			return i;
	}

	return 0;
}

void Usage()
{
	printf("Usage: pstree [ -A ] [ -n ] [ -p ] [ -U ] [ -V ]\n");
	printf("       pstree -V\n");
	printf("Display a tree of processes.\n");
	printf("  -A, --ascii         Use ASCII line drawing characters\n");
	printf("  -n, --numeric-sort  Sort output by PID\n");
	printf("  -p, --show-pids     Show PIDs; implies -c\n");
	printf("  -U, --unicode       Use UTF-8 line drawing characters\n");
	printf("  -V, --version       Display version information\n");
}

// find process with pid
struct process *find_process(pid_t pid, struct process *proc)
{
	if (proc == NULL) {
		proc = &root;
	}

	if (proc->pid == pid) {
		return proc;
	}

	struct process *res = NULL;
	if (proc->child) {
		res = find_process(pid, proc->child);
		if (res) {
			return res;
		}
	}

	if (proc->next) {
		res = find_process(pid, proc->next);
		if (res) {
			return res;
		}
	}
	return NULL;
}

void add_process(struct process *proc)
{
	struct process *self = find_process(proc->pid, NULL);
	if (self) {
		// already exists
		return;
	}

	struct process *parent = find_process(proc->ppid, NULL);
	if (parent) {
		proc->parent = parent;
		struct process *child = parent->child;
		if (!child) {
			parent->child = proc;
		} else {
			if (enable_numeric_sort) {
				if (proc->pid < child->pid) {
					proc->next = child;
					parent->child = proc;
				} else {
					while (child->next != NULL &&
					       child->next->pid < proc->pid) {
						child = child->next;
					}
					proc->next = child->next;
					child->next = proc;
				}
			} else {
				// sort by name
				// printf("%s <--> %s\n", proc->name, child->name);
				if (strcmp(proc->name, child->name) <= 0) {
					proc->next = child;
					parent->child = proc;
				} else {
					struct process *prev = NULL;
					while (child->next != NULL &&
					       strcmp(proc->name, child->name) >
						       0) {
						prev = child;
						child = child->next;
					}
					if (strcmp(proc->name, child->name) <=
					    0) {
						if (prev != NULL) {
							prev->next = proc;
							proc->next = child;
						} else {
							proc->next = child;
							parent->child = proc;
						}
					} else {
						proc->next = child->next;
						child->next = proc;
					}
				}
			}
		}
	}
}

struct process *read_process(char *pid_str, struct process *parent)
{
	char stat_file[64] = { 0 };
	if (!parent) {
		sprintf(stat_file, "/proc/%.12s/stat", pid_str);
	} else {
		sprintf(stat_file, "/proc/%d/task/%.12s/stat", parent->pid,
			pid_str);
	}

	FILE *fp = fopen(stat_file, "r");
	if (fp) {
		struct process *proc =
			(struct process *)malloc(sizeof(struct process));
		fscanf(fp, "%d (%16[^)]) %c %d", &proc->pid, proc->name,
		       &proc->state, &proc->ppid);
		proc->parent = proc->child = proc->next = NULL;
		if (parent) {
			proc->ppid = parent->pid;
		}
		add_process(proc);
		fclose(fp);
		return proc;
	}
	return NULL;
}

void concat_process_pid(struct process *proc)
{
	char pid_str[32] = "";
	sprintf(pid_str, "(%d)", proc->pid);
	strncat(proc->name, pid_str, 14);
}

void print_parent_processes(struct process *proc)
{
	if (proc->parent) {
		print_parent_processes(proc->parent);
	}

	if (enable_ascii) {
		printf("%s%*s",
		       (proc == &root ? "" : (proc->next ? " | " : "   ")),
		       (int)strlen(proc->name), "");
	} else if (enable_utf8) {
		wprintf(L"%s%*s",
			(proc == &root ? "" : (proc->next ? " │ " : "   ")),
			(int)strlen(proc->name), "");
	}
}

void print_process(struct process *proc)
{
	if (enable_show_pids) {
		concat_process_pid(proc);
	}
	if (enable_ascii) {
		printf("%s%s%s",
		       (proc == &root ? "" :
					(proc == proc->parent->child ?
						 (proc->next ? "-+-" : "---") :
						 (proc->next ? " |-" : " `-"))),
		       proc->name, proc->child ? "" : "\n");
	} else if (enable_utf8) {
		wprintf(L"%s%s%s",
			(proc == &root ?
				 "" :
				 (proc == proc->parent->child ?
					  (proc->next ? "─┬─" : "───") :
					  (proc->next ? " ├─" : " └─"))),
			proc->name, proc->child ? "" : "\n");
	}

	if (proc->child)
		print_process(proc->child);
	if (proc->next) {
		if (proc->next->parent) {
			print_parent_processes(proc->next->parent);
		}
		print_process(proc->next);
	}
}

int print_ps_tree()
{
	DIR *dir = opendir("/proc");
	if (!dir) {
		printf("failed to open dir /proc!\n");
		return -1;
	}
	struct dirent *dp;
	while ((dp = readdir(dir)) != NULL) {
		if (isdigit(*(dp->d_name))) {
			struct process *parent = read_process(dp->d_name, NULL);
			if (!parent) {
				continue;
			}
			char task_folder[64] = { 0 };
			sprintf(task_folder, "/proc/%.16s/task", dp->d_name);
			DIR *task_dir = opendir(task_folder);
			if (task_dir) {
				struct dirent *child_ptr;
				while ((child_ptr = readdir(task_dir)) !=
				       NULL) {
					if (isdigit(*(child_ptr->d_name))) {
						read_process(child_ptr->d_name,
							     parent);
					}
				}
			}
		}
	}
	closedir(dir);
	print_process(&root);
	return 0;
}

void print_version()
{
	printf("pstree v1.0\n");
	printf("author: Zhenyu PAN\n");
	printf("student id: 120090196\n");
}

int main(int argc, char *argv[])
{
	// default is using utf8 line drawing characters
	enable_utf8 = true;
	setlocale(LC_ALL, "C.UTF-8");

	int err_option_idx = 0;
	if ((err_option_idx = get_opt(argc, argv)) != 0) {
		printf("pstree: invalid option \"%s\".\n",
		       argv[err_option_idx]);
		Usage();
		return -1;
	}

	if (enable_version) {
		print_version();
	} else if (enable_ascii) {
		enable_ascii = true;
		enable_utf8 = false;
		return print_ps_tree();
	} else if (enable_utf8) {
		enable_utf8 = true;
		enable_ascii = false;
		return print_ps_tree();
	} else {
		return print_ps_tree();
	}
}
