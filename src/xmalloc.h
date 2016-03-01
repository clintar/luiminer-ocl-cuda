#pragma once

/* $OpenBSD: xmalloc.h,v 1.14 2013/05/17 00:13:14 djm Exp $ */

/*
 * Author: Tatu Ylonen <ylo@cs.hut.fi>
 * Copyright (c) 1995 Tatu Ylonen <ylo@cs.hut.fi>, Espoo, Finland
 *                    All rights reserved
 * Created: Mon Mar 20 22:09:17 1995 ylo
 *
 * Versions of malloc and friends that check their results, and never return
 * failure (they call fatal if they encounter an error).
 *
 * As far as I am concerned, the code I have written for this software
 * can be used freely for any purpose.  Any derived versions of this
 * software must be clearly marked as such, and if the derived work is
 * incompatible with the protocol description in the RFC file, it must be
 * called by a name other than "ssh" or "Secure Shell".
 */

extern "C" 
	void	*xmalloc(size_t);
extern "C"
	void	*xcalloc(size_t, size_t);
extern "C"
	void	*xrealloc(void *, size_t, size_t);
extern "C"
	char	*xstrdup(const char *);
extern "C"
	int	 xasprintf(char **, const char *, ...)
                /*__attribute__((__format__ (printf, 2, 3)))
                __attribute__((__nonnull__ (2)))*/;
extern "C"
	void     fatal(const char *, ...) /*__attribute__((noreturn))
        __attribute__((format(printf, 1, 2)))*/;

