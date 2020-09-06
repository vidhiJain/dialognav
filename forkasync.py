#!/usr/bin/env python

import sys, os, time, termios, tty, signal


# Define some custom exceptions we can raise in signal handlers
class SkipYear(Exception):
    pass



class SkipMonth(Exception):
    pass


# Process one month
def process_month(year, month):

    # Fake up whatever the processing actually is
    print 'Processing %04d-%02d' % (year, month)
    time.sleep(1)


# Process one year
def process_year(year):

    # Iterate months 1-12
    for month in range(1, 13):

        try:
            process_month(year, month)
        except SkipMonth:
            print 'Skipping month %d' % month


# Do all processing
def process_all(args):

    # Help
    print 'Started processing - args = %r' % args

    try:

        # Iterate years 2010-2015
        for year in range(2010, 2016):

            try:
                process_year(year)
            except SkipYear:
                # pass
                print 'Skipping year %d' % year
                print 'SkipYear now!'
                instruction = raw_input()
                print '>> Human: %s' % instruction

    # Handle SIGINT from parent process
    except KeyboardInterrupt:
        print 'Child caught SIGINT'

    # Return success
    print 'Child terminated normally'
    return 0


# Main entry point
def main(args):

    # Help
    print 'Press Y to skip current year, M to skip current month, or CTRL-C to abort'

    # Get file descriptor for stdin. This is almost always zero.
    stdin_fd = sys.stdin.fileno()

    # Fork here
    pid = os.fork()

    # If we're the child
    if not pid:

        # Detach child from controlling TTY, so it can't be the foreground
        # process, and therefore can't get any signals from the TTY.
        os.setsid()

        # Define signal handler for SIGUSR1 and SIGUSR2
        def on_signal(signum, frame):
            if signum == signal.SIGUSR1:
                raise SkipYear
            elif signum == signal.SIGUSR2:
                raise SkipMonth

        # We want to catch SIGUSR1 and SIGUSR2
        signal.signal(signal.SIGUSR1, on_signal)
        # signal.signal(signal.SIGUSR2, on_signal)

        # Now do the thing
        return process_all(args[1:])

    # If we get this far, we're the parent

    # Define a signal handler for when the child terminates
    def on_sigchld(signum, frame):
        assert signum == signal.SIGCHLD
        print 'Child terminated - terminating parent'
        sys.exit(0)

    # We want to catch SIGCHLD
    signal.signal(signal.SIGCHLD, on_sigchld)

    # Remember the original terminal attributes
    stdin_attrs = termios.tcgetattr(stdin_fd)

    # Change to cbreak mode, so we can detect single keypresses
    tty.setcbreak(stdin_fd)

    try:

        # Loop until we get a signal. Typically one of...
        #
        # a) SIGCHLD, when the child process terminates
        # b) SIGINT, when the user presses CTRL-C
        while 1:

            # Wait for a keypress
            char = os.read(stdin_fd, 1)
            # if char == 'y':
            #     os.kill(pid, signal.SIGUSR1)
            # # If it was 'Y', send SIGUSR1 to the child
            if char.lower() == 'y':
                os.kill(pid, signal.SIGUSR1)

            # # If it was 'M', send SIGUSR2 to the child
            if char.lower() == 'm':
                os.kill(pid, signal.SIGUSR2)

    # Parent caught SIGINT - send SIGINT to child process
    except KeyboardInterrupt:
        print 'Forwarding SIGINT to child process'
        os.kill(pid, signal.SIGINT)

    # Catch system exit
    except SystemExit:
        print 'Caught SystemExit'

    # Ensure we reset terminal attributes to original settings
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, stdin_attrs)

    # Return success
    print 'Parent terminated normally'
    return 0


# Stub
if __name__ == '__main__':
    sys.exit(main(sys.argv))
