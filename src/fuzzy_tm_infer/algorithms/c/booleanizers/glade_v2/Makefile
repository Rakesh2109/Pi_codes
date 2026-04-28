CC ?= gcc
CPPFLAGS ?= -I.
BASE_CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
ARCHFLAGS ?= -march=native
CFLAGS ?= $(BASE_CFLAGS) $(ARCHFLAGS)

.PHONY: all lib profile-lib clean

all: lib

lib: libglade_v2.so

libglade_v2.so: glade_v2.c glade_v2.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared glade_v2.c -o $@

profile-lib: glade_v2.c glade_v2.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -DGLADE_V2_PROFILE -fPIC -shared glade_v2.c -o libglade_v2.so

clean:
	rm -f libglade_v2.so
