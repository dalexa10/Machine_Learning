all: all-redirect

BASENAME=depth
LATEX = pdflatex --shell-escape
FIGURES := $(shell ls Figures/*.pdf ) #Figures/*.png)
#SUBTALKS := $(shell ls subtalk-*.tex)

.SUFFIXES: .pdf .tex

.tex.pdf: 
	$(LATEX) $*

all-redirect: talk-$(BASENAME).pdf

talk-$(BASENAME).pdf: talk-$(BASENAME).tex $(SUBTALKS) $(FIGURES)

clean:
	rm -fr _mint* *.vrb *.tex~ *.toc *.aux *.log *.nav *.out *.snm *.flc

realclean: clean
	rm -f talk-$(BASENAME).pdf

