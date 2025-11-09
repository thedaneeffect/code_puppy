"""Centralized user input utility using Rich.

This module provides a consistent way to prompt users for input across
the entire codebase, ensuring compatibility with Rich output and proper
terminal state management.
"""

import time
from typing import Optional

from rich.console import Console


def prompt_user_input(
    prompt: str = "",
    default: str = "",
    allow_empty: bool = True,
    multiline: bool = False,
    console: Optional[Console] = None,
) -> str:
    """Prompt the user for input using Rich Console.

    This function provides a centralized, consistent way to get user input
    that plays nicely with Rich output and doesn't brick the terminal.

    Args:
        prompt: The prompt message to display to the user
        default: Default value if user just presses Enter
        allow_empty: Whether to allow empty input (default True)
        multiline: Whether to allow multiline input (default False)
        console: Optional Rich Console instance (creates new one if None)

    Returns:
        The user's input as a string, or default if empty and allowed

    Raises:
        KeyboardInterrupt: If user cancels with Ctrl+C
        EOFError: If input stream ends unexpectedly
    """
    # Lazy import to avoid circular dependency with command_runner
    from code_puppy.tools.command_runner import set_awaiting_user_input

    # Use provided console or create a new one
    if console is None:
        console = Console()

    # Set the flag to indicate we're awaiting user input
    # This pauses spinners and other background activity
    set_awaiting_user_input(True)

    # Actually pause all spinners to prevent escape codes from leaking into input
    from code_puppy.messaging.spinner import pause_all_spinners, resume_all_spinners

    pause_all_spinners()

    try:
        # Give spinners time to fully stop and clear their output
        # This is critical to prevent escape sequences from bleeding into stdin
        time.sleep(0.3)

        # Flush stdin to clear any escape sequences that leaked into the buffer
        # This prevents ESC keys and cursor position codes from being read as input
        import sys
        import termios

        try:
            # On Unix-like systems, flush the input buffer
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except (AttributeError, ImportError):
            # On Windows or if termios not available, skip flushing
            pass

        # Use Rich's console.input() - it manages terminal state properly!
        if multiline:
            # For multiline, collect lines until empty line
            if prompt:
                console.print(prompt)
            console.print("[dim](Enter empty line to finish)[/dim]")
            lines = []
            while True:
                try:
                    line = console.input()
                    if not line:
                        break
                    lines.append(line)
                except EOFError:
                    break
            user_input = "\n".join(lines)
        else:
            # Simple single-line input using Rich's console.input()
            # This properly handles terminal state and supports Rich markup!
            user_input = console.input(prompt).strip()

        # Handle empty input
        if not user_input:
            if allow_empty:
                return default if default else ""
            else:
                # Re-prompt if empty not allowed
                console.print("[yellow]Input cannot be empty.[/yellow]")
                return prompt_user_input(
                    prompt=prompt,
                    default=default,
                    allow_empty=False,
                    multiline=multiline,
                    console=console,
                )

        return user_input

    except (KeyboardInterrupt, EOFError) as e:
        # Let the caller handle interruption
        raise e

    finally:
        # Always clear the awaiting input flag and resume spinners
        set_awaiting_user_input(False)
        resume_all_spinners()


def prompt_yes_no(
    prompt: str,
    default: Optional[bool] = None,
    enter_means_yes: bool = True,
    console: Optional[Console] = None,
) -> bool:
    """Prompt the user for a yes/no confirmation.

    Args:
        prompt: The prompt message to display
        default: Default value if user just presses Enter (True/False/None)
        enter_means_yes: If True, empty input (Enter) counts as yes (default True)
        console: Optional Rich Console instance (creates new one if None)

    Returns:
        True if user confirms (y/yes/enter), False otherwise

    Raises:
        KeyboardInterrupt: If user cancels with Ctrl+C
        EOFError: If input stream ends unexpectedly
    """
    # Use provided console or create a new one
    if console is None:
        console = Console()

    # Build the prompt suffix based on default
    if default is True or (default is None and enter_means_yes):
        suffix = " [bold yellow](Y/n):[/bold yellow] "
    elif default is False or (default is None and not enter_means_yes):
        suffix = " [bold yellow](y/N):[/bold yellow] "
    else:
        suffix = " [bold yellow](y/n):[/bold yellow] "

    full_prompt = prompt.rstrip() + suffix

    try:
        response = prompt_user_input(
            prompt=full_prompt,
            default="",
            allow_empty=True,
            multiline=False,
            console=console,
        )

        # Handle empty input
        if not response:
            if enter_means_yes:
                return True
            elif default is not None:
                return default
            else:
                return False

        # Check for yes/no
        response_lower = response.lower()
        if response_lower in {"yes", "y"}:
            return True
        elif response_lower in {"no", "n"}:
            return False
        else:
            # Invalid input, re-prompt with styled message
            console.print(
                f"[bold red]⚠️  Invalid input:[/bold red] [yellow]{response}[/yellow] [dim](please enter y or n)[/dim]"
            )
            return prompt_yes_no(prompt, default, enter_means_yes, console)

    except (KeyboardInterrupt, EOFError):
        # User cancelled, treat as "no"
        return False
