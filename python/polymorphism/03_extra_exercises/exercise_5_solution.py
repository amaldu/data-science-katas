



###----------------------- Exercise 5 ----------------------###


# Create a polymorphic HTML renderer. 
# Each element (Paragraph, Image, Link) should implement a method render() returning valid HTML. 
# Paragraph uses attribute "text", Image uses "src", Link uses "href" and "label"

# Create a function render_page(elements) that concatenates all HTML

# Hint: methods have to return:
# - Paragraph = <p>{self.text}</p>"
# - Image = <img src='{self.src}' />
# - Link = <a href='{self.href}'>{self.label}</a>


class HTMLElement:
    def render(self):
        raise NotImplementedError



class Paragraph(HTMLElement):
    def __init__(self, text):
        self.text = text

    def render(self):
        return f"<p>{self.text}</p>"



class Image(HTMLElement):
    def __init__(self, src):
        self.src = src

    def render(self):
        return f"<img src='{self.src}' />"



class Link(HTMLElement):
    def __init__(self, href, label):
        self.href = href
        self.label = label

    def render(self):
        return f"<a href='{self.href}'>{self.label}</a>"

def render_page(elements):
    html = ""
    for e in elements:
        html += e.render() + "\n"
    return html

# Test
elements = [
    Paragraph("Welcome to my page."),
    Image("elephant_image.jpg"),
    Link("https://google.com", "Google")
]
print(render_page(elements))


"""
<p>Welcome to my page.</p>
<img src='elephant_image.jpg' />
<a href='https://google.com'>Google</a>
"""