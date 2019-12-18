#include <iostream>
#include <string>

using std::cout;
using std::endl;
using std::string;


class Edge{
	private: string s1, s2;
	public:
	Edge(string s1, string s2){
		this->s1 = s1;
		this->s2 = s2;
	}
	
	void print(){
		cout << s1 << "->" << s2;
	}
};

class MultiEdge{
private: 
    Edge * edge;
    MultiEdge * next;
	
public:
    MultiEdge(Edge * edge, MultiEdge * next){
	    this->edge = edge;
		this->next = next;
	}
	
	virtual void print(){
		edge->print();
		next->print();
	}
};


class DiGraph{
  private: MultiEdge * edge;
  public:
	DiGraph(MultiEdge * edge){
		this->edge = edge;
	}
	
	void print(){
		edge->print();
	}
};


class EmptyEdge : public MultiEdge{
  public:
    EmptyEdge() : MultiEdge(NULL, NULL){ };
	
	void print(){ }
};

int main()
{
  DiGraph * res = new DiGraph(new MultiEdge(new Edge("abc","def"),
                                        new MultiEdge(new Edge("abc","h"), 
                                                  new EmptyEdge())));
  res->print();
}