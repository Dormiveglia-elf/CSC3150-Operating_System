.PHONY: all clean

SRCS   := $(shell find . -maxdepth 1 -name "*.c")
DEPS   := $(shell find . -maxdepth 1 -name "*.h") $(SRCS)
CFLAGS += -O0 -Wall -Werror -Wno-unused-result -Wno-unused-value -Wno-unused-variable

all: ${DEPS}
	gcc ${CFLAGS} ${SRCS} -o pstree
clean:
	rm pstree