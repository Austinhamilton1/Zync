const std = @import("std");

/// Print a long line to the debug output.
pub fn long_line() void {
    std.debug.print("-" ** 100, .{});
    std.debug.print("\n", .{});
}

/// Print a short line to the debug output.
pub fn short_line() void {
    std.debug.print("-" ** 50, .{});
    std.debug.print("\n", .{});
}

/// Print a test header to the debug output.
pub fn header(title: []const u8) void {
    std.debug.print("{s}\n", .{title});
    long_line();
}
