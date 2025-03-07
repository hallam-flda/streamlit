import streamlit as st
import inspect

st.title("Python Learning Journal")

st.header("Introduction")

st.write(
    """
    A space to add any useful tricks that I would otherwise forget.
    """
)

st.header("Friday 7th March 2025")
st.subheader("Rendering Code In Streamlit")

def add_two(n):
    return n+2

x = add_two(n=5)

with st.expander("Example Function"):
    st.code("""
            def add_two(n):
                return n+2

            x = add_two(n=5)
            """)



st.markdown(
"""
So far I have found 3 ways to render and execute code in streamlit:
"""
)

st.subheader("Option 1")
st.markdown("""
Write code blocks as text and then add the text to an expander with an execution command calling the code if it needs to be executed
"""
)

st.code("""
code_block_1 = '''
def add_two(n):
    return n+2

x = add_two(n=5)
st.write(x)        
'''
        
st.code(code_block_1)

exec(code_block_1)                
""")

st.markdown(
"""
This is probably the worst way of doing it as VS Code does not recognise the variables being defined within the text variable leading to lots of error messaging in the IDE.
It also means coming up with names for every section of code which is not always easy. Finally it also prints the streamlit commands as well if we need to display the output.
"""
)

st.subheader("Example")
code_block_1 = '''
def add_two(n):
    return n+2

x = add_two(n=5)
st.write(x)        
'''

st.code(code_block_1)
exec(code_block_1) 

st.subheader("Option 2")
st.markdown("""
Option 2 is using a combination of st.code() and st.echo()
"""
)

st.code("""
with st.echo():
    def add_two(n):
        return n+2

    x = add_two(n=5)

    st.write(x)
""")

st.subheader("Example")
with st.echo():
    def add_two(n):
        return n+2

    x = add_two(n=5)
    
    st.write(x)

st.markdown("""
This is a good option but it depends on explicitly defining the code within the echo statement every time, reducing readability. We also still see the st.write() command
"""
)

st.subheader("Option 3")
st.markdown("""
Finally my preferred option is to define functions outside the scope of any with statement directly into the .py file and use inspect to get the source of the function
""")

st.code("""
import inspect

def add_two(n):
    return n+2
              
x = add_two(5)
        
st.code(inspect.getsource(add_two)+ "\\nx = add_two(5)")
st.write(x)

""")

st.subheader("Example")

st.code(inspect.getsource(add_two)+ "\nx = add_two(5)")
st.write(x)

st.markdown("""
This approach allows for more customisability with how much of the code to output, including the st.write() or st.pyplot() functions.
            """)



