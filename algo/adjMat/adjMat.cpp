#include <iostream>
#include <vector>
#include <queue>
using namespace std;

class vertex {
    public:
        string title;
        vertex(string name) {
            title = name;
        }
};

class weightGraph{
    private:
        static const int NULL_EDGE = 0;
        vector<vertex*> vertices;
        vector<bool> marks;
        int numVertices;
        int maxVertices;
        vector< vector<int> > edges;
    public:
        weightGraph(int size){
            numVertices = 0;
            maxVertices = size;

            vertices.resize(size);
            for (int i = 0; i < size; i++){
                vertices[i] = NULL;
            }

            marks.resize(size);

            int rows = size;
            int columns = size;

            edges.resize(rows, vector<int>(columns, 0));
        }

        bool is_empty(){
            return numVertices == 0;
        };

        bool is_full(){
            return numVertices == maxVertices;
        }

        int index_is(vertex* aVertex){
            int i = 0;
            while (i < numVertices){
                if (vertices[i] == aVertex){
                    return i;
                }
                i++;
            }
            return -1;
        }

        int weight_is(int fromVertex, int toVertex){
            int row;
            int column;
            row = index_is(vertices[fromVertex]);
            column = index_is(vertices[toVertex]);
            return edges[row][column];
        }

        void addEdge(int fromVertex, int toVertex, int weight){
            int row;
            int column;
            row = index_is(vertices[fromVertex]);
            column = index_is(vertices[toVertex]);
            edges[row][column] = weight;
        }

        void addVertex(vertex* aVertex){
            vertices[numVertices] = aVertex;
            for (int i=0; i<maxVertices; i++){
                edges[numVertices][i] = NULL_EDGE;
                edges[i][numVertices] = NULL_EDGE;
            }
            numVertices++;
        }

        void clearMarks(){
            for (int i=0; i < maxVertices; i++){
                marks[i] = false;
            }
        }

        void markVertex(vertex* aVertex){
            int ix = index_is(aVertex);
            marks[ix] = true;
        }

        bool isMarked(vertex* aVertex){
            int ix = index_is(aVertex);
            return marks[ix];
        }

        vertex* getUnmarked(){
            for (int i=0; i < numVertices; i++){
                if (marks[i] == false){
                    return vertices[i];
                }
            }
        }

        void DFS(vertex* aVertex){
            int ix, ix2;
            if (aVertex != NULL) {
                cout << aVertex->title << " ";
                ix = index_is(aVertex);
                marks[ix] = true;

                for (int i = 0; i < numVertices; i++) {
                    ix2 = index_is(vertices[i]);
                    if (edges[ix][ix2] != NULL_EDGE) {
                        if (marks[i] == NULL) {
                            DFS(vertices[i]);
                        }
                    }
                }
            }
        }

        void BFS(vertex* aVertex){
            int ix, ix2;
            queue <vertex*> que;
            ix = index_is(aVertex);
            marks[ix] = true;
            que.push(aVertex);

            while (!que.empty()){
                vertex* node = que.front();
                que.pop();
                ix = index_is(node);
                cout << node -> title << " ";
                for (int i=0; i<numVertices; i++){
                    ix2 = index_is(vertices[i]);
                    if (edges[ix][ix2] != NULL_EDGE){
                        if (!marks[i]){
                            marks[i] = true;
                            que.push(vertices[i]);
                        }
                    }
                }
            }
        }

        ~weightGraph(){
            for (int i=0; i<numVertices; i++){
                delete vertices[i];
            }
        }
};

int main() {
    weightGraph AdjMatrixGraph(10);
    vertex* root;
    vertex* pVertex;


    /* create the following graph in memory, position of the * represents the direction of edges
       e.g  Edges are as following in the graph represented in the Adjacency Matrix A->B, A->C, B->D, D->C
                (A)
               /   \
              *     *
             (B)   (C)
              \     *
               *   /
                (D)
    */

    // Add vertices in memory
    root = new vertex("A");			// 0
    AdjMatrixGraph.addVertex(root);
    pVertex = new vertex("B");		// 1
    AdjMatrixGraph.addVertex(pVertex);
    pVertex = new vertex("C");		// 2
    AdjMatrixGraph.addVertex(pVertex);
    pVertex = new vertex("D");		// 3
    AdjMatrixGraph.addVertex(pVertex);

    // Add edges into memory
    AdjMatrixGraph.addEdge(0,1,1);
    AdjMatrixGraph.addEdge(0,2,1);
    AdjMatrixGraph.addEdge(1,3,1);
    AdjMatrixGraph.addEdge(3,2,1);

    // Print Depth first Search Graph Traversal
    AdjMatrixGraph.clearMarks();
    AdjMatrixGraph.DFS(root);

    cout << endl;

    // Print BFS Graph Traversal
    AdjMatrixGraph.clearMarks();
    AdjMatrixGraph.BFS(root);
    return 0;
}