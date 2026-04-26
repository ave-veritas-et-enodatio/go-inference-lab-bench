package archdiagram

import "strings"

// capitalizeASCII uppercases the first byte of s if it is a lowercase ASCII letter.
func capitalizeASCII(s string) string {
	if len(s) == 0 || s[0] < 'a' || s[0] > 'z' {
		return s
	}
	return string(s[0]-('a'-'A')) + s[1:]
}

// xmlEsc escapes &, <, > for use in SVG text content.
func xmlEsc(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	return s
}
