# Smooth_Like_Butter

## Project Overview

**Smooth_Like_Butter** is a project that helps users discover and create playlists where songs transition seamlessly. Users can select a starting song, find other tracks that flow into it, create transition chains, focus on specific genres, and generate a curated playlist.

## Key Features

1. **Song Selection**: Choose a starting track.
2. **Transition Finder**: Identify songs that transition smoothly.
3. **Chain Creation**: Build a series of smoothly transitioning tracks.
4. **Genre Focus**: Filter transitions by genre.
5. **Playlist Generation**: Create playlists with seamless transitions.

## Tools and Technologies

- **Languages**: Python
- **Data Management**: JSON
- **Visualization**: Seaborn, Matplotlib
- **Analysis**: SciKitLearn, Pandas

## Development Timeline

- **08/16/2023**: Setup environment and started collaboration on GitHub.
- **08/17/2023**: Created spectrograms, installed Seaborn, and reviewed project syntax.
- **08/21/2023**: Visualized and integrated the project roadmap into GitHub.
- **11/21/2023**: Developed API integration for song analysis.
- **11/22/2023**: Enhanced code for user input and JSON analysis.
- **12/10/2023**: Implemented tensor creation and comparison.
- **12/21/2023**: Tested Random Forest model with promising early results.

- # Project Timeline Overview

This project focuses on building a scalable pipeline to manage song similarity relationships, incorporating graph data, machine learning, and database optimizations. Below is the breakdown of tasks and phases with corresponding technologies and timelines.

## Timeline Breakdown

| **Phase**                     | **Tasks**                                              | **Technologies/Tools**              | **Estimated Time** |
|-------------------------------|--------------------------------------------------------|-------------------------------------|--------------------|
| **Phase 1: Planning**          | Define requirements, database design, and tools        | Graph design, Neo4j, API docs       | 1-2 days           |
| **Phase 2: Data Ingestion & Graph Setup** | Load song IDs, query APIs, create graph relationships | Luigi, API requests, Neo4j, NetworkX | 3-4 days           |
| **Phase 3: ML Model Integration** | Implement pre-trained ML model to enrich song data  | PyTorch/TensorFlow                  | 2-3 days           |
| **Phase 4: Path Optimization & Queries** | Implement Dijkstra’s algorithm to find optimal paths | Neo4j, Python (NetworkX)            | 2-3 days           |
| **Phase 5: Database Setup & Storage** | Store paths and results for future queries        | PostgreSQL/Redis                    | 2 days             |
| **Phase 6: Testing & Monitoring** | Test pipeline for edge cases, set up alerts and metrics | Prometheus + Grafana                | 1-2 days           |
| **Phase 7: Final Integration & Deployment** | Orchestrate tasks with Luigi, deploy on cloud infrastructure | Luigi, Docker, AWS/GCP | 3-4 days |

